import json

def is_binary(a):
    return ((a==0) | (a==1)).all()

def custom_json_dump(obj, indent=2):
    if isinstance(obj, dict):
        output = "{\n"
        comma = ""
        for key, value in obj.items():
            output += f"{comma}{' ' * (indent + 4)}{json.dumps(key)}: {custom_json_dump(value, indent + 4)}"
            comma = ",\n"
        output += f"\n{' ' * indent}}}"
        return output
    elif isinstance(obj, list):
        output = "["
        comma = ""
        for item in obj:
            output += f"{comma}{custom_json_dump(item, indent)}"
            comma = ", "
        output += "]"
        return output
    else:
        return json.dumps(obj)