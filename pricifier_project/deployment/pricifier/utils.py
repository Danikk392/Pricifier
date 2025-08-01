def format_amenities_from_string(amenities_str):
    if not isinstance(amenities_str, str):
        raise ValueError("Input must be a comma-separated string of amenities.")
    
    amenities_list = [a.strip() for a in amenities_str.split(',') if a.strip()]
    return '{' + ','.join(f'"{a}"' for a in amenities_list) + '}'
