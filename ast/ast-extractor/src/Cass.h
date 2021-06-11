#ifndef CASS_H
#define CASS_H

#include <iostream>
#include <memory>
#include <string>
#include <vector>

class Cass;
using CassVec = std::vector<std::unique_ptr<Cass>>;

class Cass {
public:

  enum class Type {
    Internal,
    NumLit,
    CharLit,
    StringLit,
    GlobalVar,
    LocalVar,
    GlobalFun,
    LocalFun,
    Error
  };

  Type type;
  std::string label;
  CassVec children;

  const Cass *prevUse = nullptr;
  const Cass *nextUse = nullptr;

  Cass(Type type, std::string label, CassVec &children);

  Cass(Type type, std::string label);

  void dump() const;

  static void serialize(const Cass *cass, std::ostream &os);

private:
  void dumpRec(int depth) const;
};

#endif
