#ifndef SEA_G2P_H
#define SEA_G2P_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Normalizes punctuation in the input string, inserting spaces before terminal
 * or separating punctuation marks (!, ?, ,, ., ;, :).
 *
 * The caller is responsible for freeing the returned string using sea_g2p_free_string.
 */
char* sea_g2p_normalize_punc(const char* input);

/**
 * Phonemizes Vietnamese/bilingual text into acoustic phoneme representations.
 *
 * The caller is responsible for freeing the returned string using sea_g2p_free_string.
 */
char* sea_g2p_phonemize(const char* input);

/**
 * Frees a string returned by sea_g2p_normalize_punc or sea_g2p_phonemize.
 * Passing NULL is a safe no-op.
 */
void sea_g2p_free_string(char* s);

/**
 * Optional: sets the path to the sea_g2p.bin dictionary.
 * Returns 0 on success, -1 on failure.
 */
int sea_g2p_set_dict_path(const char* path);

#ifdef __cplusplus
}
#endif

#endif /* SEA_G2P_H */
