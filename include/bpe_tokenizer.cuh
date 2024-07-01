#ifndef BPE_TOKENIZER_CUH
#define BPE_TOKENIZER_CUH


#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>
#include <memory>
#include <algorithm>

#define MAX_TOKEN_LENGTH 64


namespace llama {

class Tokenizer {

public:
    Tokenizer(const std::string& tokenizer_path, int vocab_size);
    ~Tokenizer() = default;

    // Disable copy constructor and assignment operator
    Tokenizer(const Tokenizer&) = delete;
    Tokenizer& operator = (const Tokenizer&) = delete;

    Tokenizer(Tokenizer&&) = default;
    Tokenizer& operator = (Tokeinzer&) = default;

    std::string decode(int prev_token, int token) const;
    void safe_printf(const std::string& piece) const;
    std::vector<int> encode(const std::string& text, bool bos, bool eos) const;

private:

    struct TokenIndex {
        std::string str;
        int id;
    }

    std::vector<std::string>> vocab;
    std::vector<std::float>> vocab_scores;
    std::vector<TokenIndex> sorted_vocab;
    int vocab_size;
    unsigned int max_token_length;
    std::array<char, 512> byte_pieces;

    void initialize_byte_pieces();
    void load_tokenizer(const std::string& tokenizer_path);
    int str_lookup(const std::string& str) const;

};
}

#endif