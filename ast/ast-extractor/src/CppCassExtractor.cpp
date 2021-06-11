#include <cassert>
#include <cstring>
#include <regex>
#include <sstream>

#include "CppCassExtractor.h"

extern "C" const TSLanguage *tree_sitter_c();
extern "C" const TSLanguage *tree_sitter_cpp();

static bool startsWith(const std::string &s1, const std::string &s2) {
  auto len1 = s1.length();
  auto len2 = s2.length();
  if (len1 < len2)
    return false;
  const char *cs1 = s1.c_str();
  const char *cs2 = s2.c_str();
  return memcmp(cs1, cs2, len2) == 0;
}

static bool endsWith(const std::string &s1, const std::string &s2) {
  auto len1 = s1.length();
  auto len2 = s2.length();
  if (len1 < len2)
    return false;
  const char *cs1 = s1.c_str() + (len1 - len2);
  const char *cs2 = s2.c_str();
  return memcmp(cs1, cs2, len2) == 0;
}

static std::string getText(const TSNode &node, const std::string &srcBytes) {
  uint32_t start_byte = ts_node_start_byte(node);
  uint32_t end_byte = ts_node_end_byte(node);
  return srcBytes.substr(start_byte, end_byte - start_byte);
}

CppCassExtractor::CppCassExtractor(bool useCParser, bool allowParseErrors)
    : CassExtractorBase(allowParseErrors) {
  if (useCParser)
    ts_parser_set_language(parser, tree_sitter_c());
  else
    ts_parser_set_language(parser, tree_sitter_cpp());
}

CppCassExtractor::~CppCassExtractor() {}

void CppCassExtractor::collectFunctions(const TSNode &node) {
  collectFunctionsRec(node);
}

void CppCassExtractor::collectFunctionsRec(const TSNode &node) {
  if (strcmp(ts_node_type(node), "function_definition") == 0) {
    TSNode body = ts_node_child_by_field_name(node, "body", 4);
    if (!ts_node_is_null(body) && ts_node_named_child_count(body) > 0)
      functions.push_back(node);
    return;
  }
  uint32_t count = ts_node_named_child_count(node);
  for (uint32_t i = 0; i < count; i++) {
    auto child = ts_node_named_child(node, i);
    collectFunctionsRec(child);
  }
}

std::unique_ptr<Cass> CppCassExtractor::buildCass(const TSNode &node) {
  auto body = ts_node_child_by_field_name(node, "body", 4);
  auto cass = buildCassRec(body);
  return cass;
}

std::unique_ptr<Cass> CppCassExtractor::buildCassRec(const TSNode &node) {
  std::string node_type(ts_node_type(node));

  // Ignore comments and preprocessing constructs
  if (node_type == "comment" || startsWith(node_type, "preproc"))
    return nullptr;

  if (ts_node_is_missing(node))
    return std::make_unique<Cass>(Cass::Type::Error, "");

  uint32_t child_count = ts_node_child_count(node);

  if (child_count == 0) {
    if (endsWith(node_type, "identifier")) {
      // Find out if it is a function name
      bool isFunctionName = false;
      TSNode parent = ts_node_parent(node);
      if (ts_node_type(parent) == std::string("call_expression")) {
        if (ts_node_child_by_field_name(parent, "function", 8).id == node.id)
          isFunctionName = true;
      } else {
        TSNode grandParent = ts_node_parent(parent);
        if (ts_node_type(grandParent) == std::string("call_expression")) {
          if (ts_node_child_by_field_name(grandParent, "function", 8).id ==
              parent.id) {
            if (ts_node_child_by_field_name(parent, "name", 4).id == node.id ||
                ts_node_child_by_field_name(parent, "field", 5).id == node.id)
              isFunctionName = true;
          }
        }
      }

      // Mark local variables
      const auto it = node2symbol.find(node.id);
      if (it == node2symbol.end()) {
        auto type =
            isFunctionName ? Cass::Type::GlobalFun : Cass::Type::GlobalVar;
        return std::make_unique<Cass>(type, getText(node, srcBytes));
      } else {
        auto type =
            isFunctionName ? Cass::Type::LocalFun : Cass::Type::LocalVar;
        auto localVarCass =
            std::make_unique<Cass>(type, getText(node, srcBytes));

        Symbol *symbol = it->second;
        const void *prevUseNodeId = symbol->getPrevUse(node.id);
        const void *nextUseNodeId = symbol->getNextUse(node.id);
        if (prevUseNodeId != nullptr) {
          auto it = node2cass.find(prevUseNodeId);
          if (it != node2cass.end()) {
            Cass *prevUse = it->second;
            localVarCass->prevUse = prevUse;
            prevUse->nextUse = localVarCass.get();
          }
        }
        if (nextUseNodeId != nullptr) {
          auto it = node2cass.find(nextUseNodeId);
          if (it != node2cass.end()) {
            Cass *nextUse = it->second;
            localVarCass->nextUse = nextUse;
            nextUse->prevUse = localVarCass.get();
          }
        }

        node2cass[node.id] = localVarCass.get();

        return localVarCass;
      }
    }

    if (node_type == "raw_string_literal") {
      auto s = getText(node, srcBytes);
      std::stringstream ss;
      for (char c : s) {
        if (c != '\t' && c != '\n' && c != '\r')
          ss << c;
      }
      s = ss.str();
      return std::make_unique<Cass>(Cass::Type::StringLit, s);
    }

    if (endsWith(node_type, "literal")) {
      return std::make_unique<Cass>(Cass::Type::NumLit,
                                    getText(node, srcBytes));
    }
  }

  // These 2 literal nodes only have 2 children: 2 (')s or 2 (")s.
  // The content is not in the parse tree.
  if (node_type == "char_literal" || node_type == "string_literal") {
    auto s = getText(node, srcBytes);
    std::stringstream ss;
    for (char c : s) {
      if (c != '\t' && c != '\n' && c != '\r')
        ss << c;
    }
    s = ss.str();

    auto type = node_type == "char_literal" ? Cass::Type::CharLit
                                            : Cass::Type::StringLit;

    return std::make_unique<Cass>(type, s);
  }

  CassVec children;
  for (uint32_t i = 0; i < child_count; i++) {
    TSNode child = ts_node_child(node, i);
    if (ts_node_is_named(child)) {
      auto cass = buildCassRec(child);
      if (cass != nullptr) {
        children.push_back(std::move(cass));
      }
    }
  }

  if (node_type == "binary_expression" || node_type == "unary_expression" ||
      node_type == "update_expression" ||
      node_type == "assignment_expression") {
    std::string op;
    for (uint32_t i = 0; i < child_count; i++) {
      bool opFound = false;
      TSNode child = ts_node_child(node, i);
      if (!ts_node_is_named(child)) {
        std::string child_type(ts_node_type(child));
        for (char c : child_type)
          if (!isalnum(c) && c != '_') {
            op = child_type;
            opFound = true;
            break;
          }
      }
      if (opFound)
        break;
    }
    node_type += op;
  }

  return std::make_unique<Cass>(Cass::Type::Internal, node_type, children);
}

void CppCassExtractor::identifyLocalVariables(const TSNode &node) {
  identifyLocalVariablesRec(node, false);
}

void CppCassExtractor::identifyLocalVariablesRec(const TSNode &node,
                                                 bool isDeclarator) {
  std::string node_type(ts_node_type(node));

  // skip scoped identifiers
  if (node_type == "scoped_identifier")
    return;

  if (node_type == "identifier") {
    // Ignore field_identifiers for now
    if (isDeclarator) {
      // Local variable declaration
      Symbol *symbol = symbolTable.set(getText(node, srcBytes));
      symbol->addUse(node.id);
      node2symbol[node.id] = symbol;
    } else {
      Symbol *symbol = symbolTable.get(getText(node, srcBytes));
      if (symbol != nullptr) {
        // Local variable reference
        symbol->addUse(node.id);
        node2symbol[node.id] = symbol;
      }
    }
    return;
  }

  if (node_type == "compound_statement" || node_type == "function_definition" ||
      node_type == "for_statement" || node_type == "for_range_loop" ||
      node_type == "if_statement" || node_type == "switch_statement" ||
      node_type == "while_statement") {
    symbolTable.enter();
    uint32_t count = ts_node_named_child_count(node);
    for (uint32_t i = 0; i < count; i++)
      identifyLocalVariablesRec(ts_node_named_child(node, i), false);
    symbolTable.exit();
    return;
  }

  if (node_type == "init_declarator") {
    TSNode valueNode = ts_node_child_by_field_name(node, "value", 5);
    if (!ts_node_is_null(valueNode))
      identifyLocalVariablesRec(valueNode, false);
    TSNode declaratorNode = ts_node_child_by_field_name(node, "declarator", 10);
    identifyLocalVariablesRec(declaratorNode, true);
    return;
  }

  TSFieldId declaratorId = ts_language_field_id_for_name(
      ts_parser_language(parser), "declarator", 10);
  TSTreeCursor cursor = ts_tree_cursor_new(node);
  if (ts_tree_cursor_goto_first_child(&cursor)) {
    do {
      TSNode child = ts_tree_cursor_current_node(&cursor);
      if (!ts_node_is_named(child))
        continue;
      TSFieldId childFieldId = ts_tree_cursor_current_field_id(&cursor);
      identifyLocalVariablesRec(child, childFieldId == declaratorId);
    } while (ts_tree_cursor_goto_next_sibling(&cursor));
  }
  ts_tree_cursor_delete(&cursor);
}
