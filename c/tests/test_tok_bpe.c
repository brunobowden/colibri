/* bpe_piece's heap-ordered merge loop against the full rescan it replaced.
 *
 * The rescan picked, after every merge, the adjacent pair with the lowest rank
 * (leftmost on a tie) by looking at all of them again: O(n^2) per pre-token
 * piece, ~40 s for one 64 KB piece. The heap must pick the same pair every time,
 * so this keeps the old loop verbatim as bpe_piece_ref and compares the two token
 * for token on random pieces, for the three tiny tokenizer fixtures, with their
 * merge lists and in the tiktoken rank-by-vocab mode (rankbpe).
 *
 * Pieces are concatenations of vocab entries' raw bytes mixed with arbitrary
 * bytes, so merge chains are long and ties are common. A 1 MB single-piece run
 * then checks that a long input completes and round-trips. */
#define _GNU_SOURCE
#include "../tok.h"

/* The pre-heap bpe_piece, verbatim. */
static void bpe_piece_ref(Tok *T, const unsigned char *p, int a, int b, int *out, int *no, int max){
    int nb=b-a;
    /* stringa byte-level (concatenazione di byte2str): <=2 byte per byte di input */
    char *s=malloc(2*nb+1); int sl=0;
    for(int i=a;i<b;i++){ int bb=p[i]; memcpy(s+sl,T->byte2str[bb],T->byte2cp_len[bb]); sl+=T->byte2cp_len[bb]; }
    s[sl]=0;
    /* ignore_merges: se l'intero pezzo e' un token, emettilo diretto */
    int whole=hm_get(&T->vocab,s,sl);
    if(whole>=0){ if(*no<max) out[(*no)++]=whole; free(s); return; }
    /* simboli iniziali = codepoint della stringa byte-level */
    int *soff=malloc((sl+1)*sizeof(int)), *slen=malloc((sl+1)*sizeof(int)); int ns=0;
    for(int i=0;i<sl;){ uint32_t cp; int k=u8_next((const unsigned char*)s,sl,i,&cp);
        soff[ns]=i; slen[ns]=k; ns++; i+=k; }
    char *kbuf=malloc(2*sl+2);
    for(;;){
        int best=INT_MAX, bp=-1;
        for(int i=0;i+1<ns;i++){
            int ll=slen[i], rl=slen[i+1];
            int rk;
            if(T->rankbpe){                                /* tiktoken: rank of the CONCATENATION */
                rk=hm_get(&T->vocab,s+soff[i],ll+rl);      /* contigui in s */
            } else {
                memcpy(kbuf,s+soff[i],ll); kbuf[ll]=0; memcpy(kbuf+ll+1,s+soff[i+1],rl);
                rk=hm_get(&T->merges,kbuf,ll+1+rl);
            }
            if(rk>=0 && rk<best){ best=rk; bp=i; }
        }
        if(bp<0) break;
        slen[bp]=soff[bp+1]+slen[bp+1]-soff[bp];          /* fonde bp e bp+1 (contigui in s) */
        for(int j=bp+1;j<ns-1;j++){ soff[j]=soff[j+1]; slen[j]=slen[j+1]; }
        ns--;
    }
    for(int i=0;i<ns;i++){
        int id=hm_get(&T->vocab,s+soff[i],slen[i]);
        if(id>=0 && *no<max) out[(*no)++]=id;
    }
    free(s); free(soff); free(slen); free(kbuf);
}

static uint64_t rng = 0x9E3779B97F4A7C15ULL;
static uint32_t rnd(void){ rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return (uint32_t)(rng >> 32); }

/* the raw bytes a vocab entry stands for (its byte-level string mapped back) */
static int raw_bytes(Tok *T, int id, unsigned char *dst, int cap){
    const char *s = T->id2str[id];
    if (!s || T->id_added[id]) return 0;
    int sl = (int)strlen(s), n = 0;
    for (int j = 0; j < sl && n < cap;) {
        uint32_t c; int k = u8_next((const unsigned char *)s, sl, j, &c); j += k;
        if (c < 1024 && T->cp2byte[c] >= 0) dst[n++] = (unsigned char)T->cp2byte[c];
    }
    return n;
}

static int fails = 0;

static void differential(const char *path, int rankbpe){
    Tok T; tok_load(&T, path);
    T.rankbpe = rankbpe;
    enum { CASES = 20000, MAXB = 400 };
    unsigned char piece[MAXB]; int got[MAXB + 1], want[MAXB + 1];
    long tokens = 0; int bad = 0;
    for (int c = 0; c < CASES; c++) {
        int n = 0, target = 1 + (int)(rnd() % 160);
        while (n < target) {
            if (rnd() % 4) {
                unsigned char tmp[64]; int k = raw_bytes(&T, (int)(rnd() % (uint32_t)T.n_ids), tmp, 64);
                if (n + k > MAXB) break;
                memcpy(piece + n, tmp, (size_t)k); n += k;
            } else {
                piece[n++] = (unsigned char)rnd();
            }
        }
        int ng = 0, nw = 0;
        bpe_piece(&T, piece, 0, n, got, &ng, MAXB + 1);
        bpe_piece_ref(&T, piece, 0, n, want, &nw, MAXB + 1);
        tokens += nw;
        if (ng != nw || memcmp(got, want, (size_t)ng * sizeof(int))) {
            if (++bad <= 3) {
                fprintf(stderr, "MISMATCH %s rankbpe=%d case %d (%d bytes): got %d ids, want %d\n",
                        path, rankbpe, c, n, ng, nw);
            }
        }
    }
    printf("test_tok_bpe: %-26s rankbpe=%d  %d pieces, %ld tokens, %d mismatch(es)\n",
           path, rankbpe, CASES, tokens, bad);
    fails += bad;
    tok_free(&T);
}

/* One piece the pre-tokenizer never splits, far past where the rescan stalled:
 * the longest all-letter vocab entry, repeated to 1 MB, so merges run along the
 * whole piece (a run of one letter would need none in a tiny vocab). */
static void long_piece(const char *path){
    Tok T; tok_load(&T, path);
    unsigned char unit[64]; int ul = 0;
    for (int id = 0; id < T.n_ids; id++) {
        unsigned char tmp[64]; int k = raw_bytes(&T, id, tmp, 64), letters = k > 0;
        for (int j = 0; j < k; j++) letters &= (tmp[j] >= 'a' && tmp[j] <= 'z');
        if (letters && k > ul) { memcpy(unit, tmp, (size_t)k); ul = k; }
    }
    int n = (1 << 20) / ul * ul;
    char *text = malloc((size_t)n), *back = malloc((size_t)n + 1);
    int *ids = malloc(sizeof(int) * (size_t)n);
    for (int i = 0; i < n; i += ul) memcpy(text + i, unit, (size_t)ul);
    int k = tok_encode(&T, text, n, ids, n);
    int m = tok_decode(&T, ids, k, back, n + 1);
    if (ul < 2 || k >= n || m != n || memcmp(back, text, (size_t)n)) {
        fprintf(stderr, "FAIL %s: %d-byte single piece of '%.*s' did not merge and round-trip "
                "(%d tokens, %d bytes back)\n", path, n, ul, (const char *)unit, k, m);
        fails++;
    } else {
        printf("test_tok_bpe: %-26s %d-byte piece of '%.*s' -> %d tokens, round-trips\n",
               path, n, ul, (const char *)unit, k);
    }
    free(text); free(back); free(ids); tok_free(&T);
}

int main(void){
    static const char *fixtures[] = {
        "tests/tok_gpt2_tiny.json", "tests/tok_o200k_tiny.json", "tests/tok_kimi_tiny.json" };
    for (int i = 0; i < 3; i++) {
        differential(fixtures[i], 0);
        differential(fixtures[i], 1);
    }
    long_piece("tests/tok_gpt2_tiny.json");
    if (fails) { fprintf(stderr, "test_tok_bpe: %d failure(s)\n", fails); return 1; }
    printf("test_tok_bpe: ok\n");
    return 0;
}
