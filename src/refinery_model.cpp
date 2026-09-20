#include "bharatopt/refinery_model.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace bharatopt {
namespace {

struct Json {
  enum class Type { Null, Bool, Number, String, Array, Object };
  Type type{Type::Null};
  bool boolean{false};
  double number{0.0};
  std::string string;
  std::vector<Json> array;
  std::unordered_map<std::string, Json> object;

  const Json& at(const std::string& key) const {
    auto it = object.find(key);
    if (it == object.end()) throw std::invalid_argument("Missing JSON field: " + key);
    return it->second;
  }
  const Json* find(const std::string& key) const {
    auto it = object.find(key);
    return it == object.end() ? nullptr : &it->second;
  }
};

class Parser {
 public:
  explicit Parser(std::string text) : text_(std::move(text)) {}
  Json parse() {
    skip();
    Json v = value();
    skip();
    if (pos_ != text_.size()) error("Trailing JSON content");
    return v;
  }

 private:
  std::string text_;
  std::size_t pos_{0};

  [[noreturn]] void error(const std::string& msg) const {
    throw std::invalid_argument("Invalid refinery JSON at byte " +
                                std::to_string(pos_) + ": " + msg);
  }
  void skip() {
    while (pos_ < text_.size() && std::isspace(static_cast<unsigned char>(text_[pos_]))) ++pos_;
  }
  char take() {
    if (pos_ >= text_.size()) error("Unexpected end of input");
    return text_[pos_++];
  }
  void expect(char c) {
    if (take() != c) error(std::string("Expected '") + c + "'");
  }
  Json value() {
    skip();
    if (pos_ >= text_.size()) error("Expected value");
    char c = text_[pos_];
    if (c == '{') return object();
    if (c == '[') return array();
    if (c == '"') return Json{Json::Type::String, false, 0.0, string()};
    if (c == 't') { literal("true"); Json v; v.type=Json::Type::Bool; v.boolean=true; return v; }
    if (c == 'f') { literal("false"); Json v; v.type=Json::Type::Bool; v.boolean=false; return v; }
    if (c == 'n') { literal("null"); Json v; v.type=Json::Type::Null; return v; }
    return number();
  }
  void literal(const char* s) {
    while (*s) {
      if (pos_ >= text_.size() || text_[pos_] != *s) error("Invalid literal");
      ++pos_; ++s;
    }
  }
  std::string string() {
    expect('"');
    std::string out;
    while (pos_ < text_.size()) {
      char c = take();
      if (c == '"') return out;
      if (c == '\\') {
        char e = take();
        switch (e) {
          case '"': case '\\': case '/': out += e; break;
          case 'b': out += '\b'; break;
          case 'f': out += '\f'; break;
          case 'n': out += '\n'; break;
          case 'r': out += '\r'; break;
          case 't': out += '\t'; break;
          default: error("Unsupported string escape");
        }
      } else out += c;
    }
    error("Unterminated string");
  }
  Json number() {
    std::size_t start = pos_;
    if (text_[pos_] == '-') ++pos_;
    while (pos_ < text_.size() && std::isdigit(static_cast<unsigned char>(text_[pos_]))) ++pos_;
    if (pos_ < text_.size() && text_[pos_] == '.') {
      ++pos_;
      while (pos_ < text_.size() && std::isdigit(static_cast<unsigned char>(text_[pos_]))) ++pos_;
    }
    if (pos_ < text_.size() && (text_[pos_] == 'e' || text_[pos_] == 'E')) {
      ++pos_;
      if (pos_ < text_.size() && (text_[pos_] == '+' || text_[pos_] == '-')) ++pos_;
      while (pos_ < text_.size() && std::isdigit(static_cast<unsigned char>(text_[pos_]))) ++pos_;
    }
    try {
      Json v; v.type=Json::Type::Number;
      v.number=std::stod(text_.substr(start, pos_-start));
      return v;
    } catch (...) { error("Invalid number"); }
  }
  Json array() {
    Json v; v.type=Json::Type::Array;
    expect('['); skip();
    if (pos_ < text_.size() && text_[pos_] == ']') { ++pos_; return v; }
    while (true) {
      v.array.push_back(value()); skip();
      char c=take();
      if (c == ']') return v;
      if (c != ',') error("Expected ',' or ']'");
    }
  }
  Json object() {
    Json v; v.type=Json::Type::Object;
    expect('{'); skip();
    if (pos_ < text_.size() && text_[pos_] == '}') { ++pos_; return v; }
    while (true) {
      skip();
      if (pos_ >= text_.size() || text_[pos_] != '"') error("Object key must be a string");
      std::string key=string(); skip(); expect(':');
      v.object.emplace(std::move(key), value()); skip();
      char c=take();
      if (c == '}') return v;
      if (c != ',') error("Expected ',' or '}'");
    }
  }
};

std::string read_all(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("Cannot open refinery JSON: " + path);
  std::ostringstream ss; ss << in.rdbuf(); return ss.str();
}
const Json& require_type(const Json& j, Json::Type t, const std::string& what) {
  if (j.type != t) throw std::invalid_argument(what + " has the wrong JSON type");
  return j;
}
std::string str(const Json& j, const std::string& what) {
  require_type(j, Json::Type::String, what); return j.string;
}
double num(const Json& j, const std::string& what) {
  require_type(j, Json::Type::Number, what);
  if (!std::isfinite(j.number)) throw std::invalid_argument(what + " must be finite");
  return j.number;
}
bool boolean(const Json& j, const std::string& what) {
  require_type(j, Json::Type::Bool, what); return j.boolean;
}
const std::vector<Json>& arr(const Json& j, const std::string& what) {
  require_type(j, Json::Type::Array, what); return j.array;
}
const std::unordered_map<std::string, Json>& obj(const Json& j, const std::string& what) {
  require_type(j, Json::Type::Object, what); return j.object;
}
RowSense sense(const std::string& s) {
  if (s=="<=" || s=="LE" || s=="le") return RowSense::LessEqual;
  if (s==">=" || s=="GE" || s=="ge") return RowSense::GreaterEqual;
  if (s=="=" || s=="EQ" || s=="eq") return RowSense::Equal;
  throw std::invalid_argument("Unknown constraint sense: " + s);
}
std::unordered_map<std::string,int> variable_index(const PoolingModel& m) {
  std::unordered_map<std::string,int> idx;
  for (std::size_t i=0;i<m.linear.var_names.size();++i) {
    if (!idx.emplace(m.linear.var_names[i], static_cast<int>(i)).second)
      throw std::invalid_argument("Duplicate variable name: " + m.linear.var_names[i]);
  }
  return idx;
}
void parse_coefficients(const Json* node,
                        const std::unordered_map<std::string,int>& idx,
                        std::vector<double>& target,
                        const std::string& what) {
  if (!node) return;
  for (const auto& [name, value] : obj(*node, what)) {
    auto it=idx.find(name);
    if (it==idx.end()) throw std::invalid_argument("Unknown variable in " + what + ": " + name);
    target[static_cast<std::size_t>(it->second)] = num(value, what + "." + name);
  }
}
std::vector<BilinearTerm> parse_bilinear(const Json* node,
                                         const std::unordered_map<std::string,int>& idx,
                                         const std::string& what) {
  std::vector<BilinearTerm> out;
  if (!node) return out;
  for (const Json& term : arr(*node, what)) {
    const auto& o=obj(term, what+" term");
    auto ix=idx.find(str(term.at("x"),what+".x"));
    auto iy=idx.find(str(term.at("y"),what+".y"));
    if (ix==idx.end() || iy==idx.end()) throw std::invalid_argument("Unknown bilinear variable in " + what);
    out.push_back({ix->second, iy->second, num(term.at("coefficient"),what+".coefficient")});
  }
  return out;
}
}

RefineryModel parse_refinery_json(const std::string& path) {
  const Json root=Parser(read_all(path)).parse();
  const auto& ro=obj(root,"root");
  RefineryModel out;
  out.pooling.name = ro.count("name") ? str(ro.at("name"),"name") : "refinery-json-model";
  out.pooling.linear.name=out.pooling.name;
  out.pooling.linear.maximize = ro.count("maximize") ? boolean(ro.at("maximize"),"maximize") : true;

  const Json& vars=root.at("variables");
  for (const Json& v : arr(vars,"variables")) {
    const auto& vo=obj(v,"variable");
    out.pooling.linear.var_names.push_back(str(v.at("name"),"variable.name"));
    out.pooling.linear.lower.push_back(num(v.at("lower"),"variable.lower"));
    out.pooling.linear.upper.push_back(num(v.at("upper"),"variable.upper"));
    out.pooling.linear.integer.push_back(false);
  }
  const std::size_t n=out.pooling.linear.var_names.size();
  out.pooling.linear.objective.assign(n,0.0);
  const auto idx=variable_index(out.pooling);
  parse_coefficients(root.find("objective_linear"),idx,out.pooling.linear.objective,"objective_linear");
  out.pooling.objective_bilinear=parse_bilinear(root.find("objective_bilinear"),idx,"objective_bilinear");

  const Json* constraints=root.find("constraints");
  if (constraints) {
    for (const Json& c : arr(*constraints,"constraints")) {
      const auto& co=obj(c,"constraint");
      const std::string name=str(c.at("name"),"constraint.name");
      const RowSense rs=sense(str(c.at("sense"),"constraint.sense"));
      const double rhs=num(c.at("rhs"),"constraint.rhs");
      out.pooling.linear.rows.push_back({name,rs,rhs});
      out.pooling.linear.row_lower.push_back(rs==RowSense::LessEqual ? -INF : rhs);
      out.pooling.linear.row_upper.push_back(rs==RowSense::GreaterEqual ? INF : rhs);
      if (rs==RowSense::Equal) {
        out.pooling.linear.row_lower.back()=rhs;
        out.pooling.linear.row_upper.back()=rhs;
      }
      out.pooling.constraint_bilinear.emplace_back();
    }
  }

  const std::size_t m=out.pooling.linear.rows.size();
  out.pooling.linear.A.rows=m; out.pooling.linear.A.cols=n;
  out.pooling.linear.A.row_ptr.assign(m+1,0);
  out.pooling.linear.A.col_index.clear(); out.pooling.linear.A.values.clear();
  out.pooling.constraint_bilinear.resize(m);
  if (constraints) {
    std::size_t row=0;
    for (const Json& c : arr(*constraints,"constraints")) {
      std::vector<double> coeff(n,0.0);
      parse_coefficients(c.find("linear"),idx,coeff,"constraint.linear");
      for (std::size_t j=0;j<n;++j) if (std::abs(coeff[j])>0.0) {
        out.pooling.linear.A.col_index.push_back(static_cast<int>(j));
        out.pooling.linear.A.values.push_back(coeff[j]);
      }
      out.pooling.linear.A.row_ptr[row+1]=out.pooling.linear.A.col_index.size();
      out.pooling.constraint_bilinear[row]=parse_bilinear(c.find("bilinear"),idx,"constraint.bilinear");
      ++row;
    }
  }
  return out;
}

void validate_refinery_model(const RefineryModel& model) {
  if (!model.pooling.valid()) throw std::invalid_argument("Refinery pooling model failed structural validation");
}

double pooling_objective_value(const PoolingModel& model,const std::vector<double>& x) {
  if (x.size()!=model.linear.A.cols) throw std::invalid_argument("Solution dimension does not match refinery model");
  double v=0.0;
  for (std::size_t j=0;j<x.size();++j) v += model.linear.objective[j]*x[j];
  for (const auto& t:model.objective_bilinear) v += t.coefficient*x[t.left]*x[t.right];
  return v;
}

double pooling_max_constraint_violation(const PoolingModel& model,const std::vector<double>& x) {
  if (x.size()!=model.linear.A.cols) throw std::invalid_argument("Solution dimension does not match refinery model");
  double worst=0.0;
  for (std::size_t j=0;j<x.size();++j) {
    worst=std::max(worst,model.linear.lower[j]-x[j]);
    worst=std::max(worst,x[j]-model.linear.upper[j]);
  }
  for (std::size_t r=0;r<model.linear.A.rows;++r) {
    double ax=0.0;
    for (std::size_t k=model.linear.A.row_ptr[r];k<model.linear.A.row_ptr[r+1];++k)
      ax += model.linear.A.values[k]*x[static_cast<std::size_t>(model.linear.A.col_index[k])];
    for (const auto& t:model.constraint_bilinear[r]) ax += t.coefficient*x[t.left]*x[t.right];
    if (std::isfinite(model.linear.row_lower[r])) worst=std::max(worst,model.linear.row_lower[r]-ax);
    if (std::isfinite(model.linear.row_upper[r])) worst=std::max(worst,ax-model.linear.row_upper[r]);
  }
  return std::max(0.0,worst);
}

std::string RefineryAuditReport::summary() const {
  std::ostringstream ss;
  ss << "Refinery Solution Audit:\n";
  ss << "  Objective: " << objective << "\n";
  ss << "  Max Bound Violation: " << max_bound_violation << "\n";
  ss << "  Max Constraint Violation: " << max_constraint_violation << "\n";
  ss << "  Status: " << (all_satisfied ? "ALL_CONSTRAINTS_SATISFIED" : "CONSTRAINTS_VIOLATED") << "\n";
  for (const auto& row : row_audits) {
    if (!row.satisfied) {
      ss << "  - Row '" << row.name << "': value=" << row.actual_value
         << " bounds=[" << row.lower << ", " << row.upper << "] violation=" << row.violation << "\n";
    }
  }
  return ss.str();
}

RefineryAuditReport audit_refinery_solution(
    const PoolingModel& model,
    const std::vector<double>& x,
    double tolerance) {
  if (x.size() != model.linear.A.cols) {
    throw std::invalid_argument("Solution dimension does not match refinery model");
  }
  RefineryAuditReport report;
  report.objective = pooling_objective_value(model, x);
  report.all_satisfied = true;

  double max_bnd = 0.0;
  for (std::size_t j = 0; j < x.size(); ++j) {
    if (std::isfinite(model.linear.lower[j])) max_bnd = std::max(max_bnd, model.linear.lower[j] - x[j]);
    if (std::isfinite(model.linear.upper[j])) max_bnd = std::max(max_bnd, x[j] - model.linear.upper[j]);
  }
  report.max_bound_violation = std::max(0.0, max_bnd);

  double max_con = 0.0;
  for (std::size_t r = 0; r < model.linear.A.rows; ++r) {
    double ax = 0.0;
    for (std::size_t k = model.linear.A.row_ptr[r]; k < model.linear.A.row_ptr[r + 1]; ++k) {
      ax += model.linear.A.values[k] * x[static_cast<std::size_t>(model.linear.A.col_index[k])];
    }
    for (const auto& t : model.constraint_bilinear[r]) {
      ax += t.coefficient * x[t.left] * x[t.right];
    }
    double viol = 0.0;
    if (std::isfinite(model.linear.row_lower[r]) && model.linear.row_lower[r] - ax > tolerance) {
      viol = model.linear.row_lower[r] - ax;
    }
    if (std::isfinite(model.linear.row_upper[r]) && ax - model.linear.row_upper[r] > tolerance) {
      viol = std::max(viol, ax - model.linear.row_upper[r]);
    }
    max_con = std::max(max_con, viol);

    ConstraintAuditDetail detail;
    detail.name = r < model.linear.rows.size() ? model.linear.rows[r].name : ("row_" + std::to_string(r));
    detail.lower = model.linear.row_lower[r];
    detail.upper = model.linear.row_upper[r];
    detail.actual_value = ax;
    detail.violation = viol;
    detail.satisfied = (viol <= tolerance);
    if (!detail.satisfied) report.all_satisfied = false;
    report.row_audits.push_back(detail);
  }
  report.max_constraint_violation = max_con;
  if (report.max_bound_violation > tolerance) report.all_satisfied = false;

  return report;
}

} // namespace bharatopt
