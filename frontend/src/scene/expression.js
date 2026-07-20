const FUNCTIONS = {
  cos: Math.cos,
  exp: Math.exp,
  ln: Math.log,
  log: Math.log,
  sec: (value) => 1 / Math.cos(value),
  sin: Math.sin,
  sqrt: Math.sqrt,
  tan: Math.tan,
};

// This is deliberately the only cache in the scene normalization layer. Its
// key is the untouched expression string (after a caller extracts it from an
// equation or overlay).
const compiledExpressions = new Map();

export class ExpressionError extends Error {
  constructor(message) {
    super(message);
    this.name = "ExpressionError";
  }
}

function tokenize(raw) {
  if (typeof raw !== "string" || !raw.trim()) {
    throw new ExpressionError("expression is empty");
  }
  if (raw.length > 512 || !/^[0-9A-Za-z+\-*/^().,\s]*$/.test(raw)) {
    throw new ExpressionError("expression contains unsupported syntax");
  }

  const tokens = [];
  let offset = 0;
  while (offset < raw.length) {
    const rest = raw.slice(offset);
    const whitespace = /^\s+/.exec(rest);
    if (whitespace) {
      offset += whitespace[0].length;
      continue;
    }

    const number = /^(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?/.exec(rest);
    if (number) {
      const value = Number(number[0]);
      if (!Number.isFinite(value)) {
        throw new ExpressionError("numeric literal is not finite");
      }
      tokens.push({ type: "number", value });
      offset += number[0].length;
      continue;
    }

    const identifier = /^[A-Za-z][A-Za-z0-9]*/.exec(rest);
    if (identifier) {
      tokens.push({ type: "identifier", value: identifier[0] });
      offset += identifier[0].length;
      continue;
    }

    if (rest.startsWith("**")) {
      tokens.push({ type: "operator", value: "^" });
      offset += 2;
      continue;
    }
    const character = rest[0];
    if ("+-*/^".includes(character)) {
      tokens.push({ type: "operator", value: character });
    } else if (character === "(" || character === ")" || character === ",") {
      tokens.push({ type: character, value: character });
    } else {
      throw new ExpressionError("expression contains unsupported syntax");
    }
    offset += 1;

    if (tokens.length > 256) {
      throw new ExpressionError("expression is too complex");
    }
  }
  tokens.push({ type: "end", value: "" });
  return tokens;
}

function parseTokens(tokens) {
  let position = 0;
  let depth = 0;

  const peek = () => tokens[position];
  const take = () => tokens[position++];
  const takeOperator = (values) => {
    const token = peek();
    if (token.type === "operator" && values.includes(token.value)) {
      take();
      return token.value;
    }
    return null;
  };

  function parseExpression() {
    let node = parseProduct();
    while (true) {
      const operator = takeOperator(["+", "-"]);
      if (!operator) return node;
      node = { type: "binary", operator, left: node, right: parseProduct() };
    }
  }

  function canMultiplyImplicitly(left, token) {
    if (token.type === "identifier") return true;
    if (token.type !== "(") return false;
    // `foo(x)` must be rejected as an unknown function, rather than silently
    // becoming multiplication. Write parameter/group products with `*`.
    return left.type !== "symbol";
  }

  function parseProduct() {
    let node = parseUnary();
    while (true) {
      const operator = takeOperator(["*", "/"]);
      if (operator) {
        node = { type: "binary", operator, left: node, right: parseUnary() };
      } else if (canMultiplyImplicitly(node, peek())) {
        node = { type: "binary", operator: "*", left: node, right: parseUnary() };
      } else {
        return node;
      }
    }
  }

  function parseUnary() {
    const operator = takeOperator(["+", "-"]);
    if (operator) {
      return { type: "unary", operator, argument: parseUnary() };
    }
    return parsePower();
  }

  function parsePower() {
    const base = parsePrimary();
    if (takeOperator(["^"])) {
      // Parsing the exponent as unary makes exponentiation right-associative
      // while retaining the conventional `-x^2 == -(x^2)` precedence.
      return { type: "binary", operator: "^", left: base, right: parseUnary() };
    }
    return base;
  }

  function parsePrimary() {
    const token = take();
    if (token.type === "number") {
      return { type: "number", value: token.value };
    }
    if (token.type === "identifier") {
      if (token.value === "pi") {
        return { type: "number", value: Math.PI };
      }
      if (Object.hasOwn(FUNCTIONS, token.value)) {
        if (peek().type !== "(") {
          throw new ExpressionError(`${token.value} requires parentheses`);
        }
        take();
        depth += 1;
        if (depth > 64) throw new ExpressionError("expression is too deeply nested");
        const argument = parseExpression();
        depth -= 1;
        if (take().type !== ")") {
          throw new ExpressionError("function call is missing a closing parenthesis");
        }
        return { type: "call", name: token.value, argument };
      }
      if (peek().type === "(") {
        throw new ExpressionError(`unknown function: ${token.value}`);
      }
      return { type: "symbol", name: token.value };
    }
    if (token.type === "(") {
      depth += 1;
      if (depth > 64) throw new ExpressionError("expression is too deeply nested");
      const expression = parseExpression();
      depth -= 1;
      if (take().type !== ")") {
        throw new ExpressionError("group is missing a closing parenthesis");
      }
      return { type: "group", argument: expression };
    }
    throw new ExpressionError("expected a number, symbol, function, or group");
  }

  const ast = parseExpression();
  if (peek().type !== "end") {
    throw new ExpressionError("unexpected trailing syntax");
  }
  return ast;
}

function collectSymbols(node, symbols) {
  if (node.type === "symbol") {
    symbols.add(node.name);
  } else if (node.type === "binary") {
    collectSymbols(node.left, symbols);
    collectSymbols(node.right, symbols);
  } else if (node.type === "unary" || node.type === "group" || node.type === "call") {
    collectSymbols(node.argument, symbols);
  }
}

function evaluateNode(node, bindings) {
  switch (node.type) {
    case "number":
      return node.value;
    case "symbol":
      return Object.hasOwn(bindings, node.name) ? bindings[node.name] : Number.NaN;
    case "group":
      return evaluateNode(node.argument, bindings);
    case "unary": {
      const value = evaluateNode(node.argument, bindings);
      return node.operator === "-" ? -value : value;
    }
    case "call":
      return FUNCTIONS[node.name](evaluateNode(node.argument, bindings));
    case "binary": {
      const left = evaluateNode(node.left, bindings);
      const right = evaluateNode(node.right, bindings);
      if (node.operator === "+") return left + right;
      if (node.operator === "-") return left - right;
      if (node.operator === "*") return left * right;
      if (node.operator === "/") return left / right;
      return left ** right;
    }
    default:
      return Number.NaN;
  }
}

export function compileExpression(raw) {
  if (compiledExpressions.has(raw)) {
    const cached = compiledExpressions.get(raw);
    if (cached instanceof ExpressionError) throw cached;
    return cached;
  }

  try {
    const ast = parseTokens(tokenize(raw));
    const symbols = new Set();
    collectSymbols(ast, symbols);
    const compiled = Object.freeze({
      raw,
      symbols: Object.freeze([...symbols].sort()),
      evaluate(bindings = {}) {
        try {
          const value = evaluateNode(ast, bindings);
          return Number.isFinite(value) ? value : Number.NaN;
        } catch {
          return Number.NaN;
        }
      },
    });
    compiledExpressions.set(raw, compiled);
    return compiled;
  } catch (error) {
    const safeError =
      error instanceof ExpressionError
        ? error
        : new ExpressionError("expression could not be parsed");
    compiledExpressions.set(raw, safeError);
    throw safeError;
  }
}

export function compilePlotEquation(raw) {
  if (typeof raw !== "string") throw new ExpressionError("plot equation is missing");
  const equals = [...raw].filter((character) => character === "=").length;
  if (equals !== 1) throw new ExpressionError("plot must contain one equation");
  const [left, right] = raw.split("=");
  if (left.trim() !== "y" || !right.trim()) {
    throw new ExpressionError("plot must have the form y = expression");
  }

  const compiled = compileExpression(right.trim());
  const parameters = compiled.symbols.filter((symbol) => symbol !== "x");
  if (
    !compiled.symbols.includes("x") ||
    parameters.length !== 1 ||
    parameters[0] === "y"
  ) {
    throw new ExpressionError("plot must use x and exactly one slider parameter");
  }
  return { compiled, parameter: parameters[0] };
}

export function clearExpressionCacheForTests() {
  compiledExpressions.clear();
}

export function expressionCacheSizeForTests() {
  return compiledExpressions.size;
}
