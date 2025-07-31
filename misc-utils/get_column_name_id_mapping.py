import requests
import json

def get_fiddler_model_column_mapping(model_id, base_url, auth_token):
    """
    Quick function to get column name to column ID mapping from Fiddler API
    Args:
        model_id: UUID of the model
        base_url: Fiddler instance URL
        auth_token: Authentication token
    Returns: Dictionary mapping column names to column IDs, or None if error
    """
    headers = {
        'Authorization': f'Bearer {auth_token}',
        'Content-Type': 'application/json',
        'Accept': 'application/json'
        }
    
    try:
        url = f"{base_url.rstrip('/')}/v3/models/{model_id}"
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        data = response.json()
        schema = data.get('data', {}).get('schema')
        
        if not schema:
            print("No schema found in response")
            return None
            
        # Create dictionary mapping column names to column IDs
        column_mapping = {}
        columns = schema.get('columns', [])
        
        for col in columns:
            name = col.get('name')
            col_id = col.get('id')
            if name and col_id:
                column_mapping[name] = col_id
        
        return column_mapping
        
    except Exception as e:
        print(f"Error: {e}")
        return None


# Quick usage example:
if __name__ == "__main__":
    # Replace these with your actual values

    URL = "https://thumbtack.fiddler.ai"
    AUTH_TOKEN = ""
    MODEL_ID = ""
    
    column_mapping = get_fiddler_model_column_mapping(MODEL_ID, URL, AUTH_TOKEN)
    
    if column_mapping:
        print(json.dumps(column_mapping, indent=2))
        print(f"\n📊 Found {len(column_mapping)} columns:")

    else:
        print("❌ Failed to get column mapping") 