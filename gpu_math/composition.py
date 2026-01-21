import string

STANDARD_HEADING = """
#version 430
layout (local_size_x = 64) in;
uniform uint nItems;
""".strip() + "\n"

def _buff_line(idx: int, name: str, instance_name: str, buffer_type: str):
    return f"layout(std430, binding = {idx}) buffer {name} {{ {buffer_type} {instance_name}[]; }};"

def create_shader(expr: str, arg_types: list[str], res_type: str) -> str:
    buffers = []
    assignments = []
    uniforms = []
    
    buffer_count = 0
    for i, (t, storage) in enumerate(arg_types):
        var_name = string.ascii_lowercase[i]
        
        if storage == 'uniform':
            uniforms.append(f"uniform {t} {var_name};")
            assignments.append(f"{t} {var_name}_val = {var_name};")
        else:
            buffers.append(_buff_line(buffer_count, f"D{i}", f"data_{i}", t))
            assignments.append(f"{t} {var_name} = data_{i}[gid];")
            buffer_count += 1
    
    if res_type != 'void':
        res_binding = buffer_count
        buffers.append(_buff_line(res_binding, "DR", "results", res_type))
    
    shader_parts = [STANDARD_HEADING]
    
    if uniforms:
        shader_parts.extend(uniforms)
        shader_parts.append("")
    
    if buffers:
        shader_parts.extend(buffers)
        shader_parts.append("")
    
    shader_parts.append(f"""
void main() {{
    uint gid = gl_GlobalInvocationID.x;
    if(gid >= nItems) return;
    
    {' '.join(assignments)}
    {f'results[gid] = {expr};' if res_type != 'void' else expr} 
}}""")
    
    return "\n".join(shader_parts)