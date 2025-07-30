import json
import os

def load_brand_info(brand_name=None, brand_filename='brand_info_new.json'):
    """Load brand information from JSON file"""
    try:
        # Load from brand_configs/brand_info_new.json
        brand_info_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'brand_configs', brand_filename)
        
        if not os.path.exists(brand_info_path):
            print(f"ERROR: brand_info_new.json not found at {brand_info_path}")
            return None
            
        with open(brand_info_path, 'r') as f:
            brand_data = json.load(f)
        
        # Load skip brands list
        skip_brands = []
        skip_brands_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'brand_configs', 'skip_brand_name.txt')
        if os.path.exists(skip_brands_path):
            with open(skip_brands_path, 'r') as f:
                skip_brands = [line.strip() for line in f.readlines() if line.strip()]
        
        # Validate structure
        if not isinstance(brand_data, dict) or 'brands' not in brand_data:
            print("ERROR: Invalid brand info JSON structure - missing 'brands' key")
            return None
            
        if not isinstance(brand_data['brands'], list):
            print("ERROR: Invalid brand info JSON structure - 'brands' should be a list")
            return None
            
        if brand_name:
            # Check if brand should be skipped
            if brand_name in skip_brands:
                print(f"INFO: Brand '{brand_name}' is in skip_brand_name.txt, skipping")
                return None
                
            # Find the specific brand
            for brand in brand_data['brands']:
                if isinstance(brand, dict) and 'name' in brand:
                    if brand['name'].lower() == brand_name.lower():
                        # Convert filepath to file_path if needed for backward compatibility
                        if 'filepath' in brand and 'file_path' not in brand:
                            brand['file_path'] = brand['filepath']
                        return brand
            print(f"WARNING: Brand '{brand_name}' not found in brand info JSON")
            return None
        else:
            # Add skip_brands to the returned data
            brand_data['skip_brands'] = skip_brands
            return brand_data
            
    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid JSON in brand info file: {e}")
        return None
    except Exception as e:
        print(f"ERROR: Failed to load brand info: {e}")
        return None