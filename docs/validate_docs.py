#!/usr/bin/env python3
"""
Documentation validation script for DiffCT documentation.

This script validates:
1. Code examples are syntactically correct
2. Internal and external links are valid
3. API docstring completeness
"""

import ast
import re
import sys
import os
import requests
import time
from pathlib import Path
from typing import List, Dict, Tuple, Set
from urllib.parse import urljoin, urlparse
import importlib.util
import inspect


class CodeValidator:
    """Validates Python code examples in documentation."""
    
    def __init__(self):
        self.errors = []
        self.warnings = []
    
    def validate_code_block(self, code: str, filename: str, line_num: int) -> bool:
        """Validate a single code block for syntax errors."""
        try:
            # Try to parse the code
            ast.parse(code)
            return True
        except SyntaxError as e:
            self.errors.append({
                'type': 'syntax_error',
                'file': filename,
                'line': line_num,
                'message': f"Syntax error in code block: {e.msg}",
                'code_line': e.lineno if e.lineno else 'unknown'
            })
            return False
        except Exception as e:
            self.warnings.append({
                'type': 'parse_warning',
                'file': filename,
                'line': line_num,
                'message': f"Could not parse code block: {str(e)}"
            })
            return False
    
    def extract_code_blocks(self, content: str, filename: str) -> List[Tuple[str, int]]:
        """Extract Python code blocks from markdown content."""
        code_blocks = []
        lines = content.split('\n')
        in_code_block = False
        current_code = []
        start_line = 0
        
        for i, line in enumerate(lines):
            if line.strip().startswith('```python'):
                in_code_block = True
                start_line = i + 1
                current_code = []
            elif line.strip() == '```' and in_code_block:
                in_code_block = False
                if current_code:
                    code_blocks.append(('\n'.join(current_code), start_line))
            elif in_code_block:
                current_code.append(line)
        
        return code_blocks
    
    def validate_file(self, filepath: Path) -> bool:
        """Validate all code blocks in a file."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            self.errors.append({
                'type': 'file_error',
                'file': str(filepath),
                'message': f"Could not read file: {str(e)}"
            })
            return False
        
        code_blocks = self.extract_code_blocks(content, str(filepath))
        all_valid = True
        
        for code, line_num in code_blocks:
            if not self.validate_code_block(code, str(filepath), line_num):
                all_valid = False
        
        return all_valid


class LinkValidator:
    """Validates internal and external links in documentation."""
    
    def __init__(self, base_path: Path):
        self.base_path = base_path
        self.errors = []
        self.warnings = []
        self.checked_urls = {}  # Cache for external URL checks
    
    def extract_links(self, content: str) -> List[Tuple[str, str]]:
        """Extract markdown links and Sphinx references."""
        links = []
        
        # Markdown links: [text](url)
        md_pattern = r'\[([^\]]+)\]\(([^)]+)\)'
        for match in re.finditer(md_pattern, content):
            links.append((match.group(1), match.group(2)))
        
        # Sphinx doc references: {doc}`path`
        doc_pattern = r'\{doc\}`([^`]+)`'
        for match in re.finditer(doc_pattern, content):
            links.append((match.group(1), match.group(1)))
        
        # Sphinx ref references: {ref}`label`
        ref_pattern = r'\{ref\}`([^`]+)`'
        for match in re.finditer(ref_pattern, content):
            links.append((match.group(1), match.group(1)))
        
        return links
    
    def validate_internal_link(self, link: str, current_file: Path) -> bool:
        """Validate internal documentation links."""
        # Handle relative paths
        if link.startswith('./') or link.startswith('../'):
            target_path = (current_file.parent / link).resolve()
        elif link.startswith('/'):
            target_path = self.base_path / link.lstrip('/')
        else:
            # Assume it's relative to docs/source
            target_path = self.base_path / 'source' / link
        
        # Add common extensions if missing
        if not target_path.suffix:
            for ext in ['.md', '.rst', '.html']:
                if (target_path.parent / (target_path.name + ext)).exists():
                    return True
        
        return target_path.exists()
    
    def validate_external_link(self, url: str) -> bool:
        """Validate external URLs with caching."""
        if url in self.checked_urls:
            return self.checked_urls[url]
        
        try:
            # Add timeout and user agent to avoid being blocked
            headers = {'User-Agent': 'Mozilla/5.0 (Documentation Validator)'}
            response = requests.head(url, timeout=10, headers=headers, allow_redirects=True)
            is_valid = response.status_code < 400
            self.checked_urls[url] = is_valid
            
            # Rate limiting to be respectful
            time.sleep(0.1)
            return is_valid
            
        except Exception as e:
            self.warnings.append({
                'type': 'external_link_warning',
                'url': url,
                'message': f"Could not validate external link: {str(e)}"
            })
            self.checked_urls[url] = False
            return False
    
    def validate_file(self, filepath: Path) -> bool:
        """Validate all links in a file."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            self.errors.append({
                'type': 'file_error',
                'file': str(filepath),
                'message': f"Could not read file: {str(e)}"
            })
            return False
        
        links = self.extract_links(content)
        all_valid = True
        
        for text, link in links:
            # Skip anchors and fragments for now
            if link.startswith('#'):
                continue
            
            # Check if it's an external URL
            if link.startswith(('http://', 'https://')):
                if not self.validate_external_link(link):
                    self.errors.append({
                        'type': 'broken_external_link',
                        'file': str(filepath),
                        'link': link,
                        'text': text,
                        'message': f"External link appears to be broken: {link}"
                    })
                    all_valid = False
            else:
                # Internal link
                if not self.validate_internal_link(link, filepath):
                    self.errors.append({
                        'type': 'broken_internal_link',
                        'file': str(filepath),
                        'link': link,
                        'text': text,
                        'message': f"Internal link not found: {link}"
                    })
                    all_valid = False
        
        return all_valid


class DocstringValidator:
    """Validates API docstring completeness."""
    
    def __init__(self):
        self.errors = []
        self.warnings = []
    
    def validate_function_docstring(self, func, module_name: str) -> bool:
        """Validate a single function's docstring."""
        if not func.__doc__:
            self.errors.append({
                'type': 'missing_docstring',
                'function': f"{module_name}.{func.__name__}",
                'message': f"Function {func.__name__} is missing a docstring"
            })
            return False
        
        docstring = func.__doc__.strip()
        
        # Check for basic components
        has_description = len(docstring.split('\n')[0]) > 10
        has_parameters = 'Parameters' in docstring or 'Args' in docstring
        has_returns = 'Returns' in docstring or 'Return' in docstring
        
        issues = []
        if not has_description:
            issues.append("missing or too short description")
        
        # Check if function has parameters
        sig = inspect.signature(func)
        if len(sig.parameters) > 0 and not has_parameters:
            issues.append("missing parameter documentation")
        
        # Check if function returns something (not None)
        if sig.return_annotation != inspect.Signature.empty and not has_returns:
            issues.append("missing return documentation")
        
        if issues:
            self.warnings.append({
                'type': 'incomplete_docstring',
                'function': f"{module_name}.{func.__name__}",
                'message': f"Docstring issues: {', '.join(issues)}"
            })
            return False
        
        return True
    
    def validate_module(self, module_path: Path) -> bool:
        """Validate all public functions in a module."""
        try:
            # Load the module
            spec = importlib.util.spec_from_file_location("module", module_path)
            if spec is None or spec.loader is None:
                self.errors.append({
                    'type': 'module_load_error',
                    'file': str(module_path),
                    'message': "Could not load module specification"
                })
                return False
            
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            all_valid = True
            
            # Check all public functions and classes
            for name in dir(module):
                if name.startswith('_'):
                    continue
                
                obj = getattr(module, name)
                
                if inspect.isfunction(obj):
                    if not self.validate_function_docstring(obj, module_path.stem):
                        all_valid = False
                elif inspect.isclass(obj):
                    # Check class docstring
                    if not obj.__doc__:
                        self.errors.append({
                            'type': 'missing_docstring',
                            'class': f"{module_path.stem}.{name}",
                            'message': f"Class {name} is missing a docstring"
                        })
                        all_valid = False
                    
                    # Check public methods
                    for method_name in dir(obj):
                        if method_name.startswith('_') and method_name != '__init__':
                            continue
                        
                        method = getattr(obj, method_name)
                        if inspect.ismethod(method) or inspect.isfunction(method):
                            if not self.validate_function_docstring(method, f"{module_path.stem}.{name}"):
                                all_valid = False
            
            return all_valid
            
        except Exception as e:
            self.errors.append({
                'type': 'module_validation_error',
                'file': str(module_path),
                'message': f"Error validating module: {str(e)}"
            })
            return False


def main():
    """Main validation function."""
    print("🔍 Starting DiffCT documentation validation...")
    
    # Get the docs directory
    docs_dir = Path(__file__).parent
    source_dir = docs_dir / 'source'
    
    # Initialize validators
    code_validator = CodeValidator()
    link_validator = LinkValidator(docs_dir)
    docstring_validator = DocstringValidator()
    
    all_passed = True
    
    # 1. Validate code examples in documentation files
    print("\n📝 Validating code examples...")
    doc_files = list(source_dir.rglob('*.md')) + list(source_dir.rglob('*.rst'))
    
    for doc_file in doc_files:
        print(f"  Checking {doc_file.relative_to(docs_dir)}...")
        if not code_validator.validate_file(doc_file):
            all_passed = False
    
    # 2. Validate links
    print("\n🔗 Validating links...")
    for doc_file in doc_files:
        print(f"  Checking links in {doc_file.relative_to(docs_dir)}...")
        if not link_validator.validate_file(doc_file):
            all_passed = False
    
    # 3. Validate API docstrings
    print("\n📚 Validating API docstrings...")
    
    # Find Python modules to validate
    project_root = docs_dir.parent
    python_files = list((project_root / 'diffct').rglob('*.py'))
    
    for py_file in python_files:
        if py_file.name.startswith('__'):
            continue
        print(f"  Checking {py_file.relative_to(project_root)}...")
        if not docstring_validator.validate_module(py_file):
            all_passed = False
    
    # Report results
    print("\n" + "="*60)
    print("📊 VALIDATION RESULTS")
    print("="*60)
    
    # Code validation results
    if code_validator.errors:
        print(f"\n❌ Code Validation Errors ({len(code_validator.errors)}):")
        for error in code_validator.errors:
            print(f"  • {error['file']}:{error.get('line', '?')} - {error['message']}")
    
    if code_validator.warnings:
        print(f"\n⚠️  Code Validation Warnings ({len(code_validator.warnings)}):")
        for warning in code_validator.warnings:
            print(f"  • {warning['file']}:{warning.get('line', '?')} - {warning['message']}")
    
    # Link validation results
    if link_validator.errors:
        print(f"\n❌ Link Validation Errors ({len(link_validator.errors)}):")
        for error in link_validator.errors:
            print(f"  • {error['file']} - {error['message']}")
    
    if link_validator.warnings:
        print(f"\n⚠️  Link Validation Warnings ({len(link_validator.warnings)}):")
        for warning in link_validator.warnings:
            print(f"  • {warning.get('file', warning.get('url', 'unknown'))} - {warning['message']}")
    
    # Docstring validation results
    if docstring_validator.errors:
        print(f"\n❌ Docstring Validation Errors ({len(docstring_validator.errors)}):")
        for error in docstring_validator.errors:
            print(f"  • {error.get('function', error.get('class', error.get('file', 'unknown')))} - {error['message']}")
    
    if docstring_validator.warnings:
        print(f"\n⚠️  Docstring Validation Warnings ({len(docstring_validator.warnings)}):")
        for warning in docstring_validator.warnings:
            print(f"  • {warning.get('function', 'unknown')} - {warning['message']}")
    
    # Summary
    total_errors = len(code_validator.errors) + len(link_validator.errors) + len(docstring_validator.errors)
    total_warnings = len(code_validator.warnings) + len(link_validator.warnings) + len(docstring_validator.warnings)
    
    print(f"\n📈 Summary:")
    print(f"  • Total errors: {total_errors}")
    print(f"  • Total warnings: {total_warnings}")
    
    if all_passed and total_errors == 0:
        print("\n✅ All validation checks passed!")
        return 0
    else:
        print(f"\n❌ Validation failed with {total_errors} errors and {total_warnings} warnings")
        return 1


if __name__ == "__main__":
    sys.exit(main())