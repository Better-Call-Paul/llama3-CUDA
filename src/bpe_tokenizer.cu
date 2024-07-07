#include "bpe_tokenizer.cuh"

namespace llama {

Tokenizer::Tokenizer(const std::string &tokenizer_path, int vocab_size) : vocab_size(vocab_size) {
    initialize_byte_pieces();
    load_tokenizer(tokenizer_path);
}

void Tokenizer::initialize_byte_pieces() {
    for (int i = 0; i < 256; i++) {
        byte_pieces[i * 2] = static_cast<char>(i);
        byte_pieces[i * 2 + 1] = '\0';
    }
}

void Tokenizer::load_tokenizer(const std::string& tokenizer_path) {
    std::ifstream file(tokenizer_path, std::ios::binary);

    if (!file) throw std::runtime_error("Failed to Load " + tokenizer_path);

    file.read(reinterpret_cast<char*>(&max_token_length), sizeof(int));

    vocab.reserve(vocab_size);
    vocab_scores.reserve(vocab_size);

    float score;
    int len;
    std::string token;
    for (int i = 0; i < vocab_size; ++i) {
        file.read(reinterpret_cast<char*>(&score), sizeof(float));
        file.read(reinterpret_cast<char*>(&len), sizeof(int));

        token.resize(len);
        file.read(&token[0], len);

        vocab.push_back(token);
        vocab_scores.push_back(score);
    }

    sorted_vocab.reserve(vocab_size);
    for (int i = 0; i < vocab_size; ++i) {
        sorted_vocab.emplace_back(TokenIndex{vocab[i], i});
    }
    std::sort(sorted_vocab.begin(), sorted_vocab.end(), [](const auto& a, const auto& b) { return a.str < b.str; });
}

int Tokenizer::str_lookup(const std::string& str) const {
    auto it = std::lower_bound(sorted_vocab.begin(), sorted_vocab.end(), str, 
                               [](const TokenIndex& a, const std::string& b) { return a.str < b; });
    if (it != sorted_vocab.end() && it->str == str) {
        return it->id;
    }
    return -1;
}

void Tokenizer::safe_printf(const std::string& piece) const {
    if (piece.empty()) return;

    unsigned char fbit = piece[0];
    if (piece.length() == 1) {
        if (!std::isprint(fbit) && !std::isspace(fbit)) {
            return;
        }
    }

    unsigned char sbit = piece.length() > 1 ? piece[1] : 0;
    unsigned char mask = 0x40;

    switch (fbit) {
        case 0xC3:
            printf("%c", sbit | mask);
            break;
        case 0xC2:
            printf("%c", sbit);
            break;
        default:
            printf("%s", piece.c_str());
    }
}

std::vector<int> Tokenizer::encode(const std::string& text, bool bos, bool eos) const {
    std::vector<int> tokens;
    
    if (bos) tokens.push_back(1);

    // Handle leading spaces
    if (!text.empty()) {
        int dummy_prefix = str_lookup(" ");
        tokens.push_back(dummy_prefix);
    }

    // Load Tokens Vector
    std::string str_buffer;
    str_buffer.reserve(max_token_length * 2 + 3);
    for (size_t i = 0; i < text.length(); ++i) {
        if ((text[i] & 0xC0) != 0x80) { // not continuation byte check 
            str_buffer.clear();
        }
        str_buffer += text[i];
        if (i + 1 == text.length() || (text[i + 1] & 0xC0) != 0x80 || str_buffer.length() >= 4) { // last byte check
            int id = str_lookup(str_buffer);
            if (id != -1) {
                tokens.push_back(id);
            }
            else {
                for (unsigned char c : str_buffer) {
                    tokens.push_back(static_cast<int>(c) + 3);
                }
            }
        }
    } 

    // BPE merging
    bool merged = true;
    while (merged) {
        merged = false;
        for (size_t i = 0; i < tokens.size() - 1; ++i) {
            str_buffer = vocab[tokens[i]] + vocab[tokens[i + 1]];
            int id = str_lookup(str_buffer);
            float curr_score = std::max(vocab_scores[tokens[i]], vocab_scores[tokens[i + 1]]);
            if (id != -1 && vocab_scores[id] > curr_score) {
                tokens[i] = id;
                tokens.erase(tokens.begin() + i + 1);
                merged = true;
                break;
            }
        }
    }

    if (eos) tokens.push_back(2);
    
    return tokens;
}

std::string Tokenizer::decode(int prev_token, int token) const {
    std::string piece = vocab[token];

    if (prev_token == 1 && piece[0] == ' ') {
        piece = piece.substr(1);
    }

    unsigned char byte_val;
    if (sscanf(piece.c_str(), "<0x%02hhX>", &byte_val) == 1) {
        piece = std::string(&byte_pieces[byte_val * 2]);
    }
    
    return piece;
}

}