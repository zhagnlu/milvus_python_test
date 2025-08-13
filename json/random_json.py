
import random
import string
import json

def random_string(length=6):
    return ''.join(random.choices(string.ascii_lowercase, k=length))

def random_value(depth=0):
    value_type = random.choice(['string', 'int', 'float', 'bool', 'null', 'object', 'array'])
    if depth > 2:
        value_type = random.choice(['string', 'int', 'float', 'bool', 'null'])  # 限制递归深度

    if value_type == 'string':
        return random_string()
    elif value_type == 'int':
        return random.randint(0, 100)
    elif value_type == 'float':
        return round(random.uniform(0, 100), 2)
    elif value_type == 'bool':
        return random.choice([True, False])
    elif value_type == 'null':
        return  1
    elif value_type == 'object':
        return random_json(depth + 1)
    elif value_type == 'array':
        return [random_value(depth + 1) for _ in range(random.randint(1, 5))]

def random_json(depth=0, max_keys=5):
    return {random_string(): random_value(depth) for _ in range(random.randint(1, max_keys))}


def generate_random_json_with_fixed_schema(key_type_map):
    def generate_value_by_type(type_name):
        if type_name == "int":
            return random.randint(-1000, 1000)
        elif type_name == "float":
            return random.uniform(-1000.0, 1000.0)
        elif type_name == "bool":
            return random.choice([True, False])
        elif type_name == "str":
            return random.choice(["a", "b", "c", "hello", "world"])
        elif type_name == "none":
            return None
        elif type_name == "list[int]":
            return random.sample(range(100), random.randint(1, 5))
        elif type_name == "list[mixed]":
            return [random.choice(["x", 1.0, 3, True]) for _ in range(3)]
        elif type_name == "dict":
            return {"nested": random.randint(1, 100)}
        elif type_name == "bigint":
            return 2**53 + random.randint(0, 100)
        elif type_name == "float64":
            return float(np.float64(random.random()))
        else:
            raise ValueError(f"Unsupported type: {type_name}")

    return {
        key: generate_value_by_type(type_name)
        for key, type_name in key_type_map.items()
    }
    
def generate_random_json(num_keys=10):
    def random_value():
        choices = [
            lambda: random.randint(-1000, 1000),                              # int
            lambda: random.uniform(-1000.0, 1000.0),                          # float
            lambda: random.choice([True, False]),                             # bool
            lambda: random.choice(["a", "b", "c", "hello", "world"]),         # str
            lambda: None,                                                    # NoneType
            lambda: random.sample(range(100), random.randint(1, 5)),         # list[int]
            lambda: [random.choice(["x", 1.0, 3, True]) for _ in range(3)],   # mixed list
            lambda: {"nested": random.randint(1, 100)},                      # nested dict
            #lambda: 2**53 + random.randint(0, 100),                          # big int
            lambda: float(np.float64(random.random())),                      # float64
        ]
        return random.choice(choices)()

    json_obj = {}
    for i in range(num_keys):
        key = f"key_{i}"
        json_obj[key] = random_value()
    return json_obj


# 生成一个随机 JSON 并打印
#test_json = random_json()
#print(json.dumps(test_json, indent=2))
