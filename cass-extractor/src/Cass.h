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
  static std::string placeholder;

  enum class Type {
    Internal,
    NumLit,
    CharLit,
    StringLit,
    GlobalVar,
    LocalVar,
    GlobalFun,
    LocalFun,
    FunSig,
    Error,
    TMP_TOKEN
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
