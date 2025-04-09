#pragma once
#include <stdio.h>
#include <iostream>
#include <map>
#include <unordered_map>
#include <vector>
#include <string>
#include <queue>

struct FileBuffer {
    FILE *f;

    FileBuffer (const std::string &fileName) {
        f = fopen(fileName.c_str(), "rb");
    }

    int ReadInt() {
        int v;
        // fread(buf,1,sizeof(buf),fp)：表示每个数据的大小为1，读了4次，一共4b，返回值为实际读取的数据个数即4
        if (fread(&v, 1, 4, f) != 4) { 
            std::cout << "FileBuffer.ReadInt error." << "\n";
        };
        return v;
    }

    float ReadFloat() {
        float v;
        if (fread(&v, 1, 4, f) != 4) {
            std::cout << "FileBuffer.ReadFloat error." << "\n";
        };
        return v;
    }

    std::string ReadString() {
        int len = ReadInt();
        std::string ret = "";
        char *v = new char[len + 5];
        v[len] = 0;
        if (fread(v, 1, len, f) != len) {
            std::cout << "FileBuffer.ReadString error." << "\n";
        }
        return v;
    }

    void ReadBytes(uint8_t *buffer, uint64_t bytes) {
        if (fread(buffer, 1, bytes, f) != bytes) {
            std::cout << "FileBuffer.ReadBytes error." << "\n";
        }
    }

    ~FileBuffer() {
        fclose(f);
    }
};

struct Tokenizer {
    struct TrieNode {
        int tokenId;
        float score;
        std::map <int, TrieNode*> next;
        TrieNode() = default;
    };
    struct Symbol {
        TrieNode *node;
        char *s;
        int pos, len;
        int prev, next;
        int fixId;

        Symbol (TrieNode *node,
                char *s, int pos, int len,
                int prev, int next, int fixId) {
            this->node = node;
            this->s = s;
            this->pos = pos;
            this->len = len;
            this->prev = prev;
            this->next = next;
            this->fixId = fixId;
        }
    };
    struct SymbolPairs {
        float score;
        int l, r, size;

        SymbolPairs(float score, int l, int r, int size) {
            this->score = score;
            this->l = l;
            this->r = r;
            this->size = size;
        }
    };

    friend bool operator < (const SymbolPairs &a, const SymbolPairs &b) {
        return a.score < b.score || (a.score == b.score && a.l > b.l);
    }

    TrieNode *root;
    std::unordered_map <int, std::string> tokenToStringDict;
    std::unordered_map <std::string, int> stringToTokenDict;
// #ifdef USE_SENTENCEPIECE
//         std::unique_ptr<sentencepiece::SentencePieceProcessor> spProcessor;
// #endif

    Tokenizer() {
        root = new TrieNode();
    }

    ~Tokenizer() {
        std::vector <TrieNode*> q;
        q.push_back(root);
        for (int i = 0; i < q.size(); i++) {
            TrieNode *now = q[i];
            for (auto it : now->next) {
                q.push_back(it.second);
            }
        }
        root = new TrieNode();
        tokenToStringDict.clear();
        delete root;
    }

    void Insert(const std::string &s, int tokenId, float score) {
        TrieNode *now = this->root;
        for (int i = 0; i < s.size(); i++) {
            if (now->next.find(s[i]) == now->next.end()) {
                now->next[s[i]] = new TrieNode();
            }
            now = now->next[s[i]];
        }
        now->tokenId = tokenId;
        now->score = score;
        tokenToStringDict[tokenId] = s;
        stringToTokenDict[s] = tokenId;
    }
    //对应于torch2flm.py
    void Initialize(std::string file){
        FileBuffer buffer(file);//这里的filename就是读取的weight文件，读了这个之后才能用tokenizer
        int versionId = buffer.ReadInt();

        if (versionId >= 1) {
            // versionId >= 1, 前置了一个key-value表
            int keyValueLen = buffer.ReadInt();
            for (int i = 0; i < keyValueLen; i++) {
                std::string key = buffer.ReadString();
                std::string value = buffer.ReadString();
                // printf("key = %s, value = %s\n", key.c_str(), value.c_str());
                // this->dicts[key] = value;
            }
        }

        // tokenizer vocab
        // bool useScore = this->dicts["tokenizer_use_score"] == "1";
        int vocabLen = buffer.ReadInt();
        for (int i = 0; i < vocabLen; i++) {
            int len = buffer.ReadInt();
            std::string x = "";
            for (int j = 0; j < len; j++) {
                x += buffer.ReadInt(); //encode内容，对应torch2flm.py#160
            }
            int id = buffer.ReadInt();
            // float score = useScore ? buffer.ReadFloat() : -i;
            float score = buffer.ReadFloat();
            Insert(x, id, score);
        }
    }
    void TryMergePairs(std::vector<Symbol> &symbols, int l, int r, std::priority_queue <SymbolPairs> &q) {
        if (l == -1 || r == -1 || symbols[l].len == 0 || symbols[r].len == 0) {
            return;
        }
        auto now = symbols[l].node;
        char *s = symbols[r].s;
        int pos = symbols[r].pos, len = symbols[r].len;
        for (int i = pos; i < pos + len; i++) {
            if (now->next.find(s[i]) != now->next.end()) {
                now = now->next[s[i]];
            } else {
                return;
            }
        }
        if (now->tokenId == -999999) {
            return;
        }
        q.push(SymbolPairs(now->score, l, r, symbols[l].len + symbols[r].len));
    } // 插入备选symbol

    std::vector<int> Encode(const std::string &ori){
        std::string blank = "";
        blank += 226, blank += 150, blank += 129;
        std::string s = blank;
        if (15 < ori.size() && ori.substr(0, 15) == "<FLM_FIX_TOKEN_") {
            s = "";
        }
        for (int i = 0; i < ori.size(); i++) {
            if (ori[i] == ' ') {
                if (i != 0 && ori[i - 1] != ' ') {
                    s += blank;
                }
            } else {
                s += ori[i];
            }
        }

        std::vector<Symbol> symbols;
        for (int i = 0; i < s.size(); i++) {
            if (i + 3 < s.size() && s[i] == '<' && s[i + 1] == 'F' && s[i + 2] == 'L' && s[i + 3] == 'M') {
                if (i + 15 < s.size() && s.substr(i, 15) == "<FLM_FIX_TOKEN_") {
                    i += 15;
                    int now = 0;
                    while (s[i] >= '0' && s[i] <= '9') {
                        now = now * 10 + s[i] - '0';
                        i++;
                    }
                    symbols.push_back(Symbol(nullptr, (char *) s.data(), i, 0, (int) symbols.size() - 1,
                                                (int) symbols.size() + 1, now));
                    continue;
                }
            }

            int tokenId = -999999, pos = i - 1;
            TrieNode *now = this->root;
            for (int j = i; j < s.size(); j++) {
                if (now->next.find(s[j]) != now->next.end()) {
                    now = now->next[s[j]];
                    if (now->tokenId != -999999) {
                        tokenId = now->tokenId;
                        pos = j;
                        break;
                    }
                } else {
                    break;
                }
            }
            if (pos >= i) {
                symbols.push_back(Symbol(now, (char *) s.data(), i, pos - i + 1, (int) symbols.size() - 1,
                                            (int) symbols.size() + 1, -999999));
                i = pos;
            } else {
                symbols.push_back(Symbol(nullptr, (char *) s.data(), i, 0, (int) symbols.size() - 1,
                                            (int) symbols.size() + 1, -999999));
            }
        }
        symbols.back().next = -1;

        std::priority_queue<SymbolPairs> workQueue;
        for (int i = 1; i < symbols.size(); i++) {
            TryMergePairs(symbols, i - 1, i, workQueue);
        }

        while (!workQueue.empty()) {
            auto top = workQueue.top();
            workQueue.pop();
            if (symbols[top.l].len == 0 || symbols[top.r].len == 0 ||
                symbols[top.l].len + symbols[top.r].len != top.size) {
                continue;
            }

            for (int i = symbols[top.r].pos; i < symbols[top.r].pos + symbols[top.r].len; i++) {
                symbols[top.l].node = symbols[top.l].node->next[symbols[top.r].s[i]];
            }
            symbols[top.l].len += symbols[top.r].len;
            symbols[top.r].len = 0;
            symbols[top.l].next = symbols[top.r].next;
            if (symbols[top.r].next >= 0) {
                symbols[symbols[top.r].next].prev = top.l;
            }

            TryMergePairs(symbols, symbols[top.l].prev, top.l, workQueue);
            TryMergePairs(symbols, top.l, symbols[top.l].next, workQueue);
        }

        std::vector<int> v;
        for (int i = 0; i < symbols.size(); i++) {
            if (symbols[i].len > 0) {
                v.push_back(symbols[i].node->tokenId);
            } else if (symbols[i].node == nullptr) {
                if (symbols[i].fixId != -999999) {
                    v.push_back(symbols[i].fixId);
                } else {
                    // 未识别的字符
                    uint8_t c = (uint8_t) (symbols[i].s[symbols[i].pos]);
                    std::string now = "<0x00>";
                    now[3] = (c / 16 > 9 ? ('A' + c / 16 - 10) : ('0' + c / 16));
                    now[4] = (c % 16 > 9 ? ('A' + c % 16 - 10) : ('0' + c % 16));
                    if (stringToTokenDict.find(now) != stringToTokenDict.end()) {
                        v.push_back(stringToTokenDict[now]);
                    }
                }
            }
        }
        return v;
    }

    // 这里的data可以换成模型的输出
    // DecodeTokens反正都是接int数组，于是我干脆把强转float给删了
    std::string Decode(std::vector<int> ret){
        std::vector <int> tokens;
        for (int i = 0; i < ret.size(); i++) {//data.Count(0)
            tokens.push_back((int)ret.data()[i]);
        }
        return DecodeTokens(tokens);
    } // 解码

    std::string DecodeTokens(const std::vector <int> &tokens){
        std::string ret = "";
        for (int i = 0; i < tokens.size(); i++) {
            std::string s = tokenToStringDict[tokens[i]];
            if (s.size() == 6 && s.substr(0, 3) == "<0x" && s.back() == '>') {
                int c = 0;
                for (int i = 3; i < 5; i++) {
                    c *= 16;
                    if (s[i] >= '0' && s[i] <= '9') {
                        c += (s[i] - '0');
                    } else {
                        c += (s[i] - 'A' + 10);
                    }
                }

                s = " ";
                s[0] = c;
            }
            if (s == "<n>") {
                ret += "\n";
            } else if (s == "<|tab|>") {
                ret += "\t";
            } else {
                ret += s;
            }
        }

        std::string blank = "";
        blank += 226, blank += 150, blank += 129;
        while (true) {
            std::string::size_type pos(0);
            if ((pos = ret.find(blank)) != std::string::npos)
                ret.replace(pos, blank.length(), " ");
            else break;
        }
        int pos = ret.find("<|blank_");
        if (pos != -1) {
            int space_num = atoi(ret.substr(8, ret.size() - 10).c_str());
            return std::string(space_num, ' ');
        }

        return ret;
    } // 解码
};