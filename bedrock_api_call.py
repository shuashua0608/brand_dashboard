import boto3
import json
import time
from botocore.exceptions import ClientError


class BedrockClaudeDistiller:
    def __init__(
        self, 
        model_id="anthropic.claude-3-sonnet-20240229-v1:0",  # Default to Claude 3 Sonnet
        region="us-east-1",  # Default to us-east-1
        aws_access_key_id=None, 
        aws_secret_access_key=None,
        max_retries=3,
        retry_delay=2,
        temperature=0,
        max_tokens=4000,
    ):
        """Initialize the AWS Bedrock client for Claude API calls"""
        from botocore.config import Config
        import os
        
        # Try to load .env file if it exists (optional)
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            # dotenv not installed, continue without it
            pass
        
        # Configure boto3 session
        session_kwargs = {"region_name": region}
        
        # Use provided credentials or fall back to environment variables
        access_key = aws_access_key_id or os.getenv('AWS_ACCESS_KEY_ID')
        secret_key = aws_secret_access_key or os.getenv('AWS_SECRET_ACCESS_KEY')
        
        if access_key and secret_key:
            session_kwargs.update({
                "aws_access_key_id": access_key,
                "aws_secret_access_key": secret_key
            })
        
        session = boto3.Session(**session_kwargs)
        
        # Create Bedrock Runtime client
        self.bedrock_runtime = session.client(
            service_name="bedrock-runtime",
            config=Config(retries={'max_attempts': max_retries, 'mode': 'standard'})
        )
        
        self.model_id = model_id
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.temperature = temperature
        self.max_tokens = max_tokens
        
    def call_claude_api(self, prompt):
        """Call Claude through Bedrock API with retries"""
        # Determine if we're using Claude 3 or earlier version based on model ID
        is_claude3 = "claude-3" in self.model_id.lower()
        
        if is_claude3:
            # Claude 3 format
            request_body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            }
        else:
            # Claude 2 format
            request_body = {
                "prompt": f"\n\nHuman: {prompt}\n\nAssistant:",
                "max_tokens_to_sample": self.max_tokens,
                "temperature": self.temperature,
                "stop_sequences": ["\n\nHuman:"]
            }
        
        for attempt in range(self.max_retries):
            try:
                response = self.bedrock_runtime.invoke_model(
                    modelId=self.model_id,
                    body=json.dumps(request_body)
                )
                
                response_body = json.loads(response["body"].read().decode("utf-8"))
                
                # Format the response consistently regardless of Claude version
                if is_claude3:
                    return response_body
                else:
                    # Convert Claude 2 response format to match Claude 3 format
                    return {
                        "content": [{
                            "text": response_body.get("completion", "")
                        }]
                    }
                
            except ClientError as e:
                if attempt < self.max_retries - 1:
                    sleep_time = self.retry_delay * (2 ** attempt)
                    print(f"API error: {e}. Retrying in {sleep_time} seconds...")
                    time.sleep(sleep_time)
                else:
                    print(f"Failed after {self.max_retries} attempts: {e}")
                    raise
        return None
    


if __name__ == "__main__":
    # TrustGPT Science credentials
    distiller = BedrockClaudeDistiller(
        model_id="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
        region="us-east-2"
    )
    
    prompt = "tell me a story"
    response = distiller.call_claude_api(prompt)
    print(response['content'][0]['text'])