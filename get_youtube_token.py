import os
import json
from google_auth_oauthlib.flow import InstalledAppFlow

def main():
    scopes = [
        "https://www.googleapis.com/auth/youtube",
        "https://www.googleapis.com/auth/youtube.upload",
        "https://www.googleapis.com/auth/youtube.readonly",
    ]
    client_secret_path = "client_secret.json"
    token_path = "token.json"

    if not os.path.exists(client_secret_path):
        print(f"Error: {client_secret_path} not found. Please ensure it is created first.")
        return

    print("Initializing OAuth Flow...")
    # Using local server flow with port 8080
    flow = InstalledAppFlow.from_client_secrets_file(
        client_secret_path, 
        scopes=scopes,
        redirect_uri="http://localhost:8080"
    )

    auth_url, _ = flow.authorization_url(prompt='consent', access_type='offline')
    
    print("\n" + "="*80)
    print("👉 GOOGLE AUTHENTICATION LINK:")
    print("="*80)
    print(auth_url)
    print("="*80 + "\n")
    print("1. Open the link above in your browser.")
    print("2. Log in with your Google account linked to your YouTube channel.")
    print("3. Grant the required permissions.")
    print("4. Your browser will redirect to http://localhost:8080/?code=xxxx...")
    print("   (It might say 'Site cannot be reached' or 'Connection Refused' - this is expected!)")
    print("5. Copy the FULL redirect URL from your browser's address bar and paste it below:")
    
    redirected_url = input("\nPaste the full URL here: ").strip()
    
    try:
        # Extract the code from the pasted URL
        from urllib.parse import urlparse, parse_qs
        parsed_url = urlparse(redirected_url)
        query_params = parse_qs(parsed_url.query)
        code = query_params.get("code")
        
        if not code:
            # Fallback if they pasted just the code
            code = [redirected_url]
            
        print("Exchanging authorization code for tokens...")
        flow.fetch_token(code=code[0])
        
        credentials = flow.credentials
        token_data = {
            "token": credentials.token,
            "refresh_token": credentials.refresh_token,
            "token_uri": credentials.token_uri,
            "client_id": credentials.client_id,
            "client_secret": credentials.client_secret,
            "scopes": credentials.scopes
        }
        
        with open(token_path, "w") as f:
            json.dump(token_data, f, indent=2)
            
        print(f"\nSUCCESS! token.json successfully created and saved.")
    except Exception as e:
        print(f"\nError exchanging code: {str(e)}")

if __name__ == "__main__":
    main()
