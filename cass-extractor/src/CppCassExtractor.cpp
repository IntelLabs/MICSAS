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

const std::unordered_set<std::string> CppCassExtractor::function_node_types = {
    "function_definition"};
const std::unordered_set<std::string> CppCassExtractor::loop_node_types = {
    "do_statement", "while_statement", "for_statement", "for_range_loop"};

CppCassExtractor::CppCassExtractor(bool useCParser, bool allowParseErrors)
    : CassExtractorBase(allowParseErrors) {
  if (useCParser)
    ts_parser_set_language(parser, tree_sitter_c());
  else
    ts_parser_set_language(parser, tree_sitter_cpp());
}

CppCassExtractor::~CppCassExtractor() {}

void CppCassExtractor::collectFunctions(const TSNode &node) {
  if (function_node_types.find(ts_node_type(node)) !=
      function_node_types.end()) {
    TSNode body = ts_node_child_by_field_name(node, "body", 4);
    if (!ts_node_is_null(body) && ts_node_named_child_count(body) > 0)
      collectedNodes.push_back(node);
    return;
  }
  uint32_t count = ts_node_named_child_count(node);
  for (uint32_t i = 0; i < count; i++) {
    auto child = ts_node_named_child(node, i);
    collectFunctions(child);
  }
}

void CppCassExtractor::collectFunctionsAndLoops(const TSNode &node) {
  std::string node_type = ts_node_type(node);
  if (function_node_types.find(node_type) != function_node_types.end()) {
    TSNode body = ts_node_child_by_field_name(node, "body", 4);
    if (!ts_node_is_null(body) && ts_node_named_child_count(body) > 0)
      collectedNodes.push_back(node);
  } else if (loop_node_types.find(node_type) != loop_node_types.end()) {
    collectedNodes.push_back(node);
  }
  uint32_t count = ts_node_named_child_count(node);
  for (uint32_t i = 0; i < count; i++) {
    auto child = ts_node_named_child(node, i);
    collectFunctionsAndLoops(child);
  }
}

std::unique_ptr<Cass> CppCassExtractor::buildCass(const TSNode &node) {
  if (function_node_types.find(ts_node_type(node)) ==
      function_node_types.end()) {
    auto cass = buildCassRec(node);
    if (cass->type == Cass::Type::TMP_TOKEN)
      return nullptr;
    return cass;
  }

  auto body = ts_node_child_by_field_name(node, "body", 4);
  auto cass = buildCassRec(body);

  if (cass->type == Cass::Type::TMP_TOKEN)
    return nullptr;

  // Function type features
  TSNode decl = node;
  std::string decl_node_type;
  do {
    if (endsWith(std::string(ts_node_type(decl)), "reference_declarator")) {
      decl = ts_node_child(decl, 1);
    } else {
      decl = ts_node_child_by_field_name(decl, "declarator", 10);
    }
    decl_node_type = ts_node_type(decl);
  } while (!endsWith(decl_node_type, "function_declarator") &&
           !endsWith(decl_node_type, "identifier"));

  auto type = ts_node_child_by_field_name(node, "type", 4);
  auto ret_type_start = ts_node_start_byte(type);
  auto ret_type_end = ts_node_start_byte(decl);
  auto ret_type =
      srcBytes.substr(ret_type_start, ret_type_end - ret_type_start);
  ret_type.erase(std::remove_if(ret_type.begin(), ret_type.end(), ::isspace),
                 ret_type.end());

  auto params = ts_node_child_by_field_name(decl, "parameters", 10);
  auto param_count =
      ts_node_is_null(params) ? 0 : ts_node_named_child_count(params);
  std::vector<std::string> param_types;
  if (param_count == 0) {
    param_types.push_back("void");
  } else {
    for (auto i = 0; i < param_count; i++) {
      auto param = ts_node_named_child(params, i);

      auto param_type = ts_node_child_by_field_name(param, "type", 4);
      auto param_type_start = ts_node_start_byte(param_type);
      auto param_type_end = ts_node_end_byte(param);

      bool hasName = true;
      TSNode param_name = param;
      do {
        param_name = ts_node_child_by_field_name(param_name, "declarator", 10);
        if (ts_node_is_null(param_name)) {
          hasName = false;
          break;
        }
      } while (std::string(ts_node_type(param_name)) != "identifier");
      auto param_name_start =
          hasName ? ts_node_start_byte(param_name) : param_type_end;
      auto param_name_end =
          hasName ? ts_node_end_byte(param_name) : param_type_end;
      auto param_type_str =
          srcBytes.substr(param_type_start,
                          param_name_start - param_type_start) +
          srcBytes.substr(param_name_end, param_type_end - param_name_end);
      param_type_str.erase(std::remove_if(param_type_str.begin(),
                                          param_type_str.end(), ::isspace),
                           param_type_str.end());

      param_types.push_back(param_type_str);
    }
  }

  int ret_card = ret_type == "void" ? 0 : 1;
  int param_card = param_types.size();
  if (param_card == 1 && param_types[0] == "void")
    param_card = 0;
  if (param_card > 2)
    param_card = 2;

  std::string sig_feat =
      "#FS#" + std::to_string(ret_card) + "_" + std::to_string(param_card);

  CassVec children;
  children.push_back(std::move(cass));
  return std::make_unique<Cass>(Cass::Type::FunSig, sig_feat, children);
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

    return std::make_unique<Cass>(Cass::Type::TMP_TOKEN,
                                  getText(node, srcBytes));
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

  std::vector<std::string> label_tokens;

  CassVec children;
  for (uint32_t i = 0; i < child_count; i++) {
    TSNode child = ts_node_child(node, i);
    if (!ts_node_is_named(child)) {
      label_tokens.push_back(std::move(getText(child, srcBytes)));
    } else {
      auto cass = buildCassRec(child);
      if (cass != nullptr) {
        if (cass->type == Cass::Type::TMP_TOKEN) {
          label_tokens.push_back(std::move(cass->label));
        } else {
          children.push_back(std::move(cass));
          label_tokens.push_back(Cass::placeholder);
        }
      }
    }
  }

  std::stringstream ss;
  for (const auto &s : label_tokens)
    ss << s;
  std::string label = ss.str();

  if (children.empty()) {
    if (label_tokens.empty())
      return nullptr;
    return std::make_unique<Cass>(Cass::Type::TMP_TOKEN, label);
  }

  if (children.size() == 1 && label_tokens.size() == 1)
    return std::move(children[0]);

  label = "#" + node_type + "#" + label;

  return std::make_unique<Cass>(Cass::Type::Internal, label, children);
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
