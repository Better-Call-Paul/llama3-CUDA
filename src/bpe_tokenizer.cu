#include "bpe_tokenizer.cuh"

namespace llama {


Tokeinzer::Tokeinzer(const std::string &tokenizer_path, int vocab_size) : vocab_size(vocab_size) {
    initialize_byte_pieces();
    load_tokenizer(tokenizer_path);
}

Tokenizer::initialize_byte_pieces() {
    for (int i = 0; i < 256; i++) {
        byte_pieces[i * 2] = static_cast<char>(i);
        byte_pieces[i * 2 + 1] = '\0';
    }
}

Tokenizer::load_tokenizer(const std::string& tokenizer_path){
    std::ifstream file(tokenizer_path, std::ios::binary);

    if (!file) throw std::runtime_error("Failed to Load " + tokenizer_path);

    file.read(reinterept_cast<char*>(&max_token_length), sizeof(int));

    vocab.reserve(vocab_size);
    vocab_scores.reserve(vocab_size);

    float score;
    int len;
    std::string token;
    for (int i = 0; i < vocab_size; ++i) {
        file.read(reinterept_cast<char*>(&score), sizeof(float));
        file.read(reinterept_cast<char*>(&len), sizeof(int));

        token.resize(len);
        file.read(&token[0], len);

        vocab.push_back(token);
        vocab_scores.push_back(score);
    }

    sorted_vocab.reserve(vocab_size);
    for (int i = 0; i < vocab_size; ++i) {
        sorted_vocab.emplace_back(vocab[i], i);
    }
    return std::sort(sorted_vocab.begin(), sorted_vocab.end(), []{const auto& a, const auto& b} { return a.str < b.str;});
}



}