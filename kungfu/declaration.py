import itertools

NUMERIC = ["int", "uint", "float"]
INTEGRAL = ["int", "uint"]
FLOATING = ["float", "double"]

def get_sigs(arity: int, types: list, res_type_override: str = None):
    sigs = []
    for combo in itertools.product(types, repeat=arity):
        if res_type_override:
            res = res_type_override
        else:
            res = "float" if "float" in combo else combo[0]
        sigs.append((list(combo), res))
    return sigs

CONFIG = {
    "full": {1: ("a", get_sigs(1, NUMERIC))},
    "neg": {1: ("-a", get_sigs(1, NUMERIC))},
    "square": {1: ("a * a", get_sigs(1, NUMERIC))},
    "is_zero": {1: ("a == 0", get_sigs(1, NUMERIC, "bool"))},
    "bool": {1: ("a != 0", get_sigs(1, NUMERIC, "bool"))},
    "bool_not": {1: ("!a", get_sigs(1, ["bool"], "bool"))},

    "add": {
        2: ("a + b", get_sigs(2, NUMERIC)),
        3: ("a + b + c", get_sigs(3, NUMERIC)),
        4: ("a + b + c + d", get_sigs(4, NUMERIC))
    },
    "sub": {2: ("a - b", get_sigs(2, NUMERIC))},
    "mult": {
        2: ("a * b", get_sigs(2, NUMERIC)),
        3: ("a * b * c", get_sigs(3, NUMERIC))
    },
    "div": {2: ("(b != 0) ? (a / b) : 0", get_sigs(2, NUMERIC))},
    "floordiv": {2: ("(b != 0) ? int(floor(float(a) / float(b))) : 0", get_sigs(2, NUMERIC, "int"))},
    "mod": {2: ("(b != 0) ? (a % b) : 0", get_sigs(2, INTEGRAL))},
    
    "avg": {
        2: ("(a + b) / 2.0", get_sigs(2, NUMERIC, "float")),
        3: ("(a + b + c) / 3.0", get_sigs(3, NUMERIC, "float")),
        4: ("(a + b + c + d) / 4.0", get_sigs(4, NUMERIC, "float"))
    },

    "and": {2: ("a & b", get_sigs(2, INTEGRAL))},
    "or": {2: ("a | b", get_sigs(2, INTEGRAL))},
    "xor": {2: ("a ^ b", get_sigs(2, INTEGRAL))},
    "lsh": {2: ("a << b", get_sigs(2, INTEGRAL))},
    "rsh": {2: ("a >> b", get_sigs(2, INTEGRAL))},
    "bitwise_not": {1: ("~a", get_sigs(1, INTEGRAL))},
    
    "gt": {2: ("a > b", get_sigs(2, NUMERIC, "bool"))},
    "lt": {2: ("a < b", get_sigs(2, NUMERIC, "bool"))},
    "eq": {2: ("a == b", get_sigs(2, NUMERIC, "bool"))},
    "gte": {2: ("a >= b", get_sigs(2, NUMERIC, "bool"))},
    "lte": {2: ("a <= b", get_sigs(2, NUMERIC, "bool"))},
    "neq": {2: ("a != b", get_sigs(2, NUMERIC, "bool"))},

    "clamp": {3: ("clamp(a, b, c)", get_sigs(3, ["float"]))},

    "sqrt": {1: ("sqrt(float(a))", get_sigs(1, NUMERIC, "float"))},
    "exp": {1: ("exp(float(a))", get_sigs(1, NUMERIC, "float"))},
    "log": {1: ("log(float(a))", get_sigs(1, NUMERIC, "float"))},
    "pow": {2: ("pow(float(a), float(b))", get_sigs(2, NUMERIC, "float"))},
    "abs": {1: ("abs(a)", get_sigs(1, NUMERIC))},

    "sin": {1: ("sin(float(a))", get_sigs(1, NUMERIC, "float"))},
    "cos": {1: ("cos(float(a))", get_sigs(1, NUMERIC, "float"))},
    "tan": {1: ("tan(float(a))", get_sigs(1, NUMERIC, "float"))},
    "asin": {1: ("asin(float(a))", get_sigs(1, NUMERIC, "float"))},
    "acos": {1: ("acos(float(a))", get_sigs(1, NUMERIC, "float"))},
    "atan": {
        1: ("atan(float(a))", get_sigs(1, NUMERIC, "float")),
        2: ("atan(float(a), float(b))", get_sigs(2, NUMERIC, "float"))
    },

    "floor": {1: ("floor(float(a))", get_sigs(1, NUMERIC, "float"))},
    "ceil": {1: ("ceil(float(a))", get_sigs(1, NUMERIC, "float"))},
    "fract": {1: ("fract(float(a))", get_sigs(1, NUMERIC, "float"))},
    "round": {1: ("round(float(a))", get_sigs(1, NUMERIC, "float"))},
    "sign": {1: ("sign(a)", get_sigs(1, NUMERIC))},
    
    "min": {2: ("min(a, b)", get_sigs(2, NUMERIC))},
    "max": {2: ("max(a, b)", get_sigs(2, NUMERIC))},
    "mix": {3: ("mix(a, b, c)", get_sigs(3, ["float"]))},
    "step": {2: ("step(a, b)", get_sigs(2, ["float"]))},
    "smoothstep": {3: ("smoothstep(a, b, c)", get_sigs(3, ["float"]))},

    # Complex number operations
    "cmul_real": {4: ("(a * c) - (b * d)", get_sigs(4, ["float"], "float"))},
    "cmul_imag": {4: ("(a * d) + (b * c)", get_sigs(4, ["float"], "float"))},
}