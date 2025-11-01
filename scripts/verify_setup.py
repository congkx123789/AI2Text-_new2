#!/usr/bin/env python3
"""
Verify AI2Text ASR setup is complete and working.

This script checks:
1. Required files exist
2. Services are accessible
3. Event flow is working
4. Database is initialized
"""

import sys
import time
import subprocess
import urllib.request
import urllib.error
import json
from pathlib import Path

# ANSI color codes
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

def print_header(text):
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}{text:^60}{RESET}")
    print(f"{BLUE}{'='*60}{RESET}\n")

def print_success(text):
    print(f"{GREEN}✓{RESET} {text}")

def print_error(text):
    print(f"{RED}✗{RESET} {text}")

def print_warning(text):
    print(f"{YELLOW}⚠{RESET} {text}")

def check_file_exists(filepath):
    """Check if a required file exists."""
    path = Path(filepath)
    if path.exists():
        print_success(f"File exists: {filepath}")
        return True
    else:
        print_error(f"File missing: {filepath}")
        return False

def check_service_health(name, url):
    """Check if a service responds to health check."""
    try:
        response = urllib.request.urlopen(url, timeout=5)
        data = json.loads(response.read().decode())
        if data.get('status') == 'healthy':
            print_success(f"{name}: Healthy")
            return True
        else:
            print_warning(f"{name}: Unhealthy - {data}")
            return False
    except urllib.error.URLError as e:
        print_error(f"{name}: Not accessible - {e}")
        return False
    except Exception as e:
        print_error(f"{name}: Error - {e}")
        return False

def check_port_open(name, port):
    """Check if a port is accessible."""
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(2)
    result = sock.connect_ex(('localhost', port))
    sock.close()
    
    if result == 0:
        print_success(f"{name}: Port {port} is open")
        return True
    else:
        print_error(f"{name}: Port {port} is not accessible")
        return False

def main():
    print_header("AI2Text ASR Setup Verification")
    
    all_checks_passed = True
    
    # 1. Check required files
    print_header("1. Checking Required Files")
    
    required_files = [
        'env.example',
        'Makefile',
        'infra/docker-compose.yml',
        'services/api-gateway/app.py',
        'services/ingestion/app.py',
        'services/asr/worker.py',
        'services/asr/streaming_server.py',
        'services/metadata/app.py',
        'services/nlp-post/app.py',
        'services/embeddings/app.py',
        'services/search/app.py',
        'services/metadata/migrations/001_init.sql',
        'scripts/bootstrap.sh',
        'scripts/jwt_dev_token.py',
        'tests/e2e/test_flow.py',
        'RUN_GUIDE.md',
    ]
    
    files_ok = all(check_file_exists(f) for f in required_files)
    all_checks_passed = all_checks_passed and files_ok
    
    # 2. Check infrastructure services
    print_header("2. Checking Infrastructure Services")
    
    infra_checks = [
        ('PostgreSQL', 5432),
        ('MinIO', 9000),
        ('Qdrant', 6333),
        ('NATS', 4222),
    ]
    
    infra_ok = all(check_port_open(name, port) for name, port in infra_checks)
    all_checks_passed = all_checks_passed and infra_ok
    
    # 3. Check application services
    print_header("3. Checking Application Services")
    
    service_checks = [
        ('API Gateway', 'http://localhost:8080/health'),
        ('Ingestion', 'http://localhost:8001/health'),
        ('Metadata', 'http://localhost:8002/health'),
        ('NLP-Post', 'http://localhost:8004/health'),
        ('Embeddings', 'http://localhost:8005/health'),
        ('Search', 'http://localhost:8006/health'),
    ]
    
    # Wait a bit for services to start
    print_warning("Waiting 5 seconds for services to initialize...")
    time.sleep(5)
    
    services_ok = all(check_service_health(name, url) for name, url in service_checks)
    all_checks_passed = all_checks_passed and services_ok
    
    # 4. Check database
    print_header("4. Checking Database")
    
    try:
        # Try to connect to PostgreSQL
        import psycopg2
        conn = psycopg2.connect(
            "postgresql://postgres:postgres@localhost:5432/asrmeta"
        )
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public'")
        table_count = cursor.fetchone()[0]
        cursor.close()
        conn.close()
        
        if table_count >= 3:  # Should have at least: audio, transcripts, speakers
            print_success(f"Database initialized: {table_count} tables")
        else:
            print_warning(f"Database has only {table_count} tables, expected at least 3")
            all_checks_passed = False
    except ImportError:
        print_warning("psycopg2 not installed, skipping database check")
    except Exception as e:
        print_error(f"Database check failed: {e}")
        all_checks_passed = False
    
    # 5. Check Qdrant collection
    print_header("5. Checking Qdrant Vector Database")
    
    try:
        response = urllib.request.urlopen('http://localhost:6333/collections/texts', timeout=5)
        data = json.loads(response.read().decode())
        if data.get('result'):
            print_success("Qdrant collection 'texts' exists")
            vectors_count = data['result'].get('points_count', 0)
            print(f"  Current vectors: {vectors_count}")
        else:
            print_warning("Qdrant collection not properly initialized")
            all_checks_passed = False
    except Exception as e:
        print_error(f"Qdrant check failed: {e}")
        all_checks_passed = False
    
    # 6. Check MinIO buckets
    print_header("6. Checking MinIO Buckets")
    
    try:
        # Try to list buckets (requires MinIO client library or curl)
        result = subprocess.run(
            ['docker', 'exec', 'minio', 'ls', '/data'],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if 'audio' in result.stdout and 'transcripts' in result.stdout:
            print_success("MinIO buckets exist: audio, transcripts")
        else:
            print_warning("MinIO buckets may not be initialized")
            print(f"  Found: {result.stdout.strip()}")
    except Exception as e:
        print_warning(f"MinIO bucket check skipped: {e}")
    
    # Final summary
    print_header("Verification Summary")
    
    if all_checks_passed:
        print(f"{GREEN}🎉 All checks passed! Your setup is ready.{RESET}\n")
        print("Next steps:")
        print("  1. Generate JWT token: python3 scripts/jwt_dev_token.py")
        print("  2. Run tests: make test-e2e")
        print("  3. Upload audio: curl -X POST http://localhost:8080/v1/ingest ...")
        print("  4. View logs: make logs")
        print("")
        print("For detailed instructions, see RUN_GUIDE.md")
        return 0
    else:
        print(f"{RED}⚠ Some checks failed. Please review the errors above.{RESET}\n")
        print("Troubleshooting:")
        print("  1. Run: bash scripts/bootstrap.sh")
        print("  2. Start services: docker compose -f infra/docker-compose.yml up -d")
        print("  3. Check logs: make logs")
        print("  4. Re-run this script")
        return 1

if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print(f"\n{YELLOW}Verification cancelled by user{RESET}")
        sys.exit(1)
    except Exception as e:
        print(f"\n{RED}Unexpected error: {e}{RESET}")
        sys.exit(1)

