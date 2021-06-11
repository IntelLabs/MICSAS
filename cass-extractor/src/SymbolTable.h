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
