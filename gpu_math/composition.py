import string

def get_standard_heading(layout=(64, 1, 1), include_n_items=True):
    """Generate standard shader heading with custom layout"""
    if isinstance(layout, (int, float)):
        layout = (int(layout), 1, 1)
    elif len(layout) == 2:
        layout = (layout[0], layout[1], 1)
    
    heading = f"""#version 430
layout (local_size_x = {layout[0]}, local_size_y = {layout[1]}, local_size_z = {layout[2]}) in;"""
    
    if include_n_items:
        heading += "\nuniform uint nItems;"
    
    return heading.strip() + "\n"

def _buff_line(idx: int, name: str, instance_name: str, buffer_type: str):
    return f"layout(std430, binding = {idx}) buffer {name} {{ {buffer_type} {instance_name}[]; }};"

def create_shader(expr: str, arg_types: list[str], res_type: str, layout=(64, 1, 1)) -> str:
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
    
    shader_parts = [get_standard_heading(layout)]
    
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