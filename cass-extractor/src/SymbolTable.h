/*
MIT License

Copyright (c) 2021 Intel Labs

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/
#ifndef SYMBOL_TABLE_H
#define SYMBOL_TABLE_H

#include <memory>
#include <unordered_map>
#include <vector>

class Symbol {
public:
  void addUse(const void *use) {
    use2id[use] = uses.size();
    uses.push_back(use);
  }
  const void *getPrevUse(const void *use) const {
    return getUseByOffset(use, -1);
  }
  const void *getNextUse(const void *use) const {
    return getUseByOffset(use, 1);
  }
  const void *getUseByOffset(const void *use, int offset) const {
    const auto it = use2id.find(use);
    if (it == use2id.end())
      return nullptr;
    int i = it->second + offset;
    if (i < 0 || i >= uses.size())
      return nullptr;
    return uses[i];
  }

private:
  std::vector<const void *> uses;
  std::unordered_map<const void *, int> use2id;
};

class SymbolTable {
public:
  SymbolTable() { tables.emplace_back(); }

  void enter() { tables.emplace_back(); }

  void exit() { tables.pop_back(); }

  Symbol *set(std::string name) {
    auto symbol = new Symbol();
    tables[tables.size() - 1][name] = symbol;
    symbols.push_back(std::unique_ptr<Symbol>(symbol));
    return symbol;
  }

  Symbol *get(std::string name) const {
    auto i = tables.size() - 1;
    while (i > 0) {
      const auto &table = tables[i];
      const auto it = table.find(name);
      if (it != table.end())
        return it->second;
      i--;
    }
    return nullptr;
  }

  void clear() {
    tables.clear();
    tables.emplace_back();
    symbols.clear();
  }

private:
  std::vector<std::unordered_map<std::string, Symbol *>> tables;
  std::vector<std::unique_ptr<Symbol>> symbols;
};

#endif
