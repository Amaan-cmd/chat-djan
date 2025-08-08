import requests
from bs4 import BeautifulSoup
import time

# Test one URL to see the HTML structure
url = "https://calamitymod.wiki.gg/wiki/Sunken_Sea"

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

# Add delay and session to handle anti-bot measures
session = requests.Session()
session.headers.update(headers)

print("Fetching page...")
response = session.get(url)
time.sleep(2)  # Wait for page to load
soup = BeautifulSoup(response.content, 'html.parser')

print("=== TESTING HTML STRUCTURE ===")
title = soup.find('title')
print(f"Page title: {title.get_text() if title else 'No title'}")
print(f"Page length: {len(response.text)} characters")
print(f"Status code: {response.status_code}")

# Look for common content containers
possible_containers = [
    'mw-parser-output',
    'mw-content-text', 
    'content',
    'main-content',
    'article-content',
    'page-content'
]

print("\n=== LOOKING FOR CONTENT CONTAINERS ===")
for container_class in possible_containers:
    element = soup.find('div', class_=container_class)
    if element:
        text = element.get_text()[:200]
        print(f"Found '{container_class}': {len(text)} chars")
        print(f"  Preview: {text[:100]}...")
    else:
        print(f"Not found: '{container_class}'")

# Also check for ID-based containers
print("\n=== CHECKING ID-BASED CONTAINERS ===")
possible_ids = ['content', 'main', 'article', 'bodyContent']
for container_id in possible_ids:
    element = soup.find('div', id=container_id)
    if element:
        text = element.get_text()[:200]
        print(f"Found ID '{container_id}': {len(text)} chars")
        print(f"  Preview: {text[:100]}...")
    else:
        print(f"Not found ID: '{container_id}'")

# Show all div classes to help identify the right one
print("\n=== ALL DIV CLASSES (first 10) ===")
divs_with_class = soup.find_all('div', class_=True)[:10]
for div in divs_with_class:
    classes = div.get('class', [])
    text_preview = div.get_text()[:50].replace('\n', ' ').strip()
    print(f"Class: {classes} -> '{text_preview}...'")