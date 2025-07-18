#!/usr/bin/env python3
"""
Simple link checker for DiffCT documentation.

This script specifically checks:
- Internal documentation links
- External URLs
- Sphinx cross-references
"""

import re
import sys
import requests
import time
from pathlib import Path
from typing import List, Tuple, Dict
from urllib.parse import urljoin, urlparse


class SimpleLinkChecker:
    """Simple but effective link checker for documentation."""
    
    def __init__(self, docs_path: Path):
        self.docs_path = docs_path
        self.source_path = docs_path / 'source'
        self.build_path = docs_path / 'build' / 'html'
        self.errors = []
        self.warnings = []
        self.external_cache = {}
    
    def find_all_links(self, content: str) -> List[Tuple[str, str, int]]:
        """Find all links in content with line numbers."""
        links = []
        lines = content.split('\n')
        
        for line_num, line in enumerate(lines, 1):
            # Markdown links: [text](url)
            md_pattern = r'\[([^\]]*)\]\(([^)]+)\)'
            for match in re.finditer(md_pattern, line):
                links.append((match.group(1), match.group(2), line_num))
            
            # Sphinx doc references: {doc}`path`
            doc_pattern = r'\{doc\}`([^`]+)`'
            for match in re.finditer(doc_pattern, line):
                links.append((f"doc:{match.group(1)}", match.group(1), line_num))
            
            # Sphinx ref references: {ref}`label`
            ref_pattern = r'\{ref\}`([^`]+)`'
            for match in re.finditer(ref_pattern, line):
                links.append((f"ref:{match.group(1)}", match.group(1), line_num))
            
            # RST links: `text <url>`_
            rst_pattern = r'`([^<]+)<([^>]+)>`_'
            for match in re.finditer(rst_pattern, line):
                links.append((match.group(1), match.group(2), line_num))
        
        return links
    
    def check_internal_link(self, link: str, current_file: Path) -> bool:
        """Check if internal link exists."""
        # Handle different types of internal links
        if link.startswith('#'):
            # Anchor link - skip for now
            return True
        
        # Handle Sphinx doc references
        if not link.startswith(('http://', 'https://', 'mailto:')):
            # Try different possible paths
            possible_paths = []
            
            # Relative to current file
            if link.startswith('./') or link.startswith('../'):
                possible_paths.append((current_file.parent / link).resolve())
            else:
                # Relative to source directory
                possible_paths.append(self.source_path / link)
                possible_paths.append(self.source_path / f"{link}.md")
                possible_paths.append(self.source_path / f"{link}.rst")
                
                # Check in subdirectories
                for subdir in ['api', 'examples']:
                    possible_paths.append(self.source_path / subdir / link)
                    possible_paths.append(self.source_path / subdir / f"{link}.md")
                    possible_paths.append(self.source_path / subdir / f"{link}.rst")
            
            # Check if any path exists
            for path in possible_paths:
                if path.exists():
                    return True
            
            # Check in built HTML
            if self.build_path.exists():
                html_path = self.build_path / f"{link}.html"
                if html_path.exists():
                    return True
        
        return False
    
    def check_external_link(self, url: str) -> bool:
        """Check if external URL is accessible."""
        if url in self.external_cache:
            return self.external_cache[url]
        
        try:
            # Clean up the URL
            if not url.startswith(('http://', 'https://')):
                return True  # Skip non-HTTP URLs
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Documentation Link Checker)',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
            }
            
            response = requests.head(url, timeout=10, headers=headers, allow_redirects=True)
            
            # Some servers don't support HEAD, try GET
            if response.status_code == 405:
                response = requests.get(url, timeout=10, headers=headers, stream=True)
            
            is_valid = response.status_code < 400
            self.external_cache[url] = is_valid
            
            # Be respectful with rate limiting
            time.sleep(0.2)
            
            return is_valid
            
        except requests.exceptions.Timeout:
            self.warnings.append(f"Timeout checking URL: {url}")
            self.external_cache[url] = False
            return False
        except requests.exceptions.RequestException as e:
            self.warnings.append(f"Error checking URL {url}: {str(e)}")
            self.external_cache[url] = False
            return False
        except Exception as e:
            self.warnings.append(f"Unexpected error checking URL {url}: {str(e)}")
            self.external_cache[url] = False
            return False
    
    def check_file(self, file_path: Path) -> Dict:
        """Check all links in a single file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            return {
                'file': str(file_path),
                'error': f"Could not read file: {str(e)}",
                'links_checked': 0,
                'broken_links': [],
                'warnings': []
            }
        
        links = self.find_all_links(content)
        broken_links = []
        file_warnings = []
        
        for text, url, line_num in links:
            # Skip empty links
            if not url.strip():
                continue
            
            # Check external links
            if url.startswith(('http://', 'https://')):
                if not self.check_external_link(url):
                    broken_links.append({
                        'type': 'external',
                        'url': url,
                        'text': text,
                        'line': line_num,
                        'message': f"External link appears broken: {url}"
                    })
            else:
                # Check internal links
                if not self.check_internal_link(url, file_path):
                    broken_links.append({
                        'type': 'internal',
                        'url': url,
                        'text': text,
                        'line': line_num,
                        'message': f"Internal link not found: {url}"
                    })
        
        return {
            'file': str(file_path),
            'links_checked': len(links),
            'broken_links': broken_links,
            'warnings': file_warnings
        }
    
    def check_all_files(self) -> Dict:
        """Check all documentation files."""
        print("🔍 Finding documentation files...")
        
        # Find all documentation files
        doc_files = []
        for pattern in ['*.md', '*.rst']:
            doc_files.extend(self.source_path.rglob(pattern))
        
        print(f"📄 Found {len(doc_files)} documentation files")
        
        results = {
            'files_checked': 0,
            'total_links': 0,
            'broken_links': 0,
            'files_with_errors': [],
            'all_warnings': []
        }
        
        for doc_file in doc_files:
            print(f"  Checking {doc_file.relative_to(self.docs_path)}...")
            
            file_result = self.check_file(doc_file)
            results['files_checked'] += 1
            results['total_links'] += file_result['links_checked']
            
            if file_result.get('error'):
                results['files_with_errors'].append(file_result)
            elif file_result['broken_links']:
                results['broken_links'] += len(file_result['broken_links'])
                results['files_with_errors'].append(file_result)
            
            if file_result['warnings']:
                results['all_warnings'].extend(file_result['warnings'])
        
        # Add global warnings
        results['all_warnings'].extend(self.warnings)
        
        return results


def main():
    """Main function."""
    print("🔗 DiffCT Documentation Link Checker")
    print("=" * 50)
    
    # Find docs directory
    docs_dir = Path(__file__).parent
    if not docs_dir.exists():
        print("❌ Could not find docs directory")
        return 1
    
    # Initialize checker
    checker = SimpleLinkChecker(docs_dir)
    
    # Run checks
    results = checker.check_all_files()
    
    # Report results
    print("\n" + "=" * 50)
    print("📊 LINK CHECK RESULTS")
    print("=" * 50)
    
    print(f"\n📈 Summary:")
    print(f"  • Files checked: {results['files_checked']}")
    print(f"  • Total links found: {results['total_links']}")
    print(f"  • Broken links: {results['broken_links']}")
    print(f"  • Files with issues: {len(results['files_with_errors'])}")
    print(f"  • Warnings: {len(results['all_warnings'])}")
    
    # Show broken links
    if results['files_with_errors']:
        print(f"\n❌ Files with broken links:")
        for file_result in results['files_with_errors']:
            if file_result.get('error'):
                print(f"\n  📄 {file_result['file']}")
                print(f"     Error: {file_result['error']}")
            else:
                print(f"\n  📄 {file_result['file']}")
                for broken in file_result['broken_links']:
                    print(f"     Line {broken['line']}: {broken['message']}")
                    if broken['text']:
                        print(f"       Text: '{broken['text']}'")
    
    # Show warnings
    if results['all_warnings']:
        print(f"\n⚠️  Warnings:")
        for warning in results['all_warnings']:
            print(f"  • {warning}")
    
    # Final status
    if results['broken_links'] == 0 and not results['files_with_errors']:
        print("\n✅ All links are valid!")
        return 0
    else:
        print(f"\n❌ Found {results['broken_links']} broken links in {len(results['files_with_errors'])} files")
        return 1


if __name__ == "__main__":
    sys.exit(main())