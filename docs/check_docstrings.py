#!/usr/bin/env python3
"""
Docstring completeness checker for DiffCT API functions.

This script checks:
1. All public functions have docstrings
2. Docstrings contain required sections (Parameters, Returns, etc.)
3. Parameter documentation matches function signatures
"""

import ast
import sys
import inspect
import importlib.util
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple


class DocstringChecker:
    """Checks docstring completeness for Python modules."""
    
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.stats = {
            'functions_checked': 0,
            'classes_checked': 0,
            'methods_checked': 0,
            'missing_docstrings': 0,
            'incomplete_docstrings': 0
        }
    
    def parse_docstring(self, docstring: str) -> Dict[str, Any]:
        """Parse docstring to extract sections."""
        if not docstring:
            return {}
        
        lines = docstring.strip().split('\n')
        sections = {
            'description': '',
            'parameters': [],
            'returns': '',
            'raises': [],
            'examples': []
        }
        
        current_section = 'description'
        description_lines = []
        
        for line in lines:
            line = line.strip()
            
            # Check for section headers
            if line.lower() in ['parameters:', 'params:', 'args:', 'arguments:']:
                current_section = 'parameters'
                continue
            elif line.lower() in ['returns:', 'return:', 'yields:', 'yield:']:
                current_section = 'returns'
                continue
            elif line.lower() in ['raises:', 'raise:', 'except:', 'exceptions:']:
                current_section = 'raises'
                continue
            elif line.lower() in ['examples:', 'example:']:
                current_section = 'examples'
                continue
            elif line.lower() in ['note:', 'notes:', 'warning:', 'warnings:']:
                current_section = 'notes'
                continue
            
            # Add content to current section
            if current_section == 'description' and line:
                description_lines.append(line)
            elif current_section == 'parameters' and line:
                sections['parameters'].append(line)
            elif current_section == 'returns' and line:
                sections['returns'] += line + ' '
            elif current_section == 'raises' and line:
                sections['raises'].append(line)
            elif current_section == 'examples' and line:
                sections['examples'].append(line)
        
        sections['description'] = ' '.join(description_lines)
        sections['returns'] = sections['returns'].strip()
        
        return sections
    
    def check_function_signature(self, func) -> Dict[str, Any]:
        """Extract function signature information."""
        try:
            sig = inspect.signature(func)
            return {
                'parameters': list(sig.parameters.keys()),
                'has_return_annotation': sig.return_annotation != inspect.Signature.empty,
                'parameter_annotations': {
                    name: param.annotation != inspect.Parameter.empty 
                    for name, param in sig.parameters.items()
                }
            }
        except (ValueError, TypeError):
            return {
                'parameters': [],
                'has_return_annotation': False,
                'parameter_annotations': {}
            }
    
    def validate_function_docstring(self, func, full_name: str) -> List[Dict]:
        """Validate a single function's docstring."""
        issues = []
        self.stats['functions_checked'] += 1
        
        # Check if docstring exists
        if not func.__doc__:
            self.stats['missing_docstrings'] += 1
            issues.append({
                'type': 'missing_docstring',
                'severity': 'error',
                'message': 'Function has no docstring'
            })
            return issues
        
        # Parse docstring
        docstring_info = self.parse_docstring(func.__doc__)
        sig_info = self.check_function_signature(func)
        
        # Check description
        if not docstring_info['description'] or len(docstring_info['description']) < 10:
            issues.append({
                'type': 'short_description',
                'severity': 'warning',
                'message': 'Description is missing or too short (< 10 characters)'
            })
        
        # Check parameters documentation
        func_params = [p for p in sig_info['parameters'] if p not in ['self', 'cls']]
        
        if func_params:
            if not docstring_info['parameters']:
                issues.append({
                    'type': 'missing_parameters',
                    'severity': 'error',
                    'message': f'Function has {len(func_params)} parameters but no parameter documentation'
                })
            else:
                # Check if all parameters are documented
                documented_params = []
                for param_line in docstring_info['parameters']:
                    # Extract parameter name (various formats)
                    param_match = None
                    for param in func_params:
                        if param in param_line:
                            documented_params.append(param)
                            break
                
                missing_params = set(func_params) - set(documented_params)
                if missing_params:
                    issues.append({
                        'type': 'undocumented_parameters',
                        'severity': 'warning',
                        'message': f'Parameters not documented: {", ".join(missing_params)}'
                    })
        
        # Check return documentation
        if sig_info['has_return_annotation'] or 'return' in func.__doc__.lower():
            if not docstring_info['returns']:
                issues.append({
                    'type': 'missing_returns',
                    'severity': 'warning',
                    'message': 'Function appears to return a value but has no return documentation'
                })
        
        # Check for examples in important functions
        if not docstring_info['examples'] and len(func_params) > 2:
            issues.append({
                'type': 'missing_examples',
                'severity': 'info',
                'message': 'Complex function would benefit from usage examples'
            })
        
        if issues:
            self.stats['incomplete_docstrings'] += 1
        
        return issues
    
    def validate_class_docstring(self, cls, full_name: str) -> List[Dict]:
        """Validate a class docstring."""
        issues = []
        self.stats['classes_checked'] += 1
        
        if not cls.__doc__:
            self.stats['missing_docstrings'] += 1
            issues.append({
                'type': 'missing_docstring',
                'severity': 'error',
                'message': 'Class has no docstring'
            })
            return issues
        
        docstring_info = self.parse_docstring(cls.__doc__)
        
        if not docstring_info['description'] or len(docstring_info['description']) < 20:
            issues.append({
                'type': 'short_description',
                'severity': 'warning',
                'message': 'Class description is missing or too short (< 20 characters)'
            })
        
        return issues
    
    def validate_method_docstring(self, method, class_name: str, method_name: str) -> List[Dict]:
        """Validate a method docstring."""
        issues = []
        self.stats['methods_checked'] += 1
        
        # Special handling for common methods
        if method_name in ['__init__', '__str__', '__repr__']:
            if not method.__doc__:
                issues.append({
                    'type': 'missing_docstring',
                    'severity': 'warning',
                    'message': f'Special method {method_name} should have a docstring'
                })
            return issues
        
        # Skip private methods
        if method_name.startswith('_') and method_name != '__init__':
            return issues
        
        # Regular method validation
        return self.validate_function_docstring(method, f"{class_name}.{method_name}")
    
    def check_module_from_file(self, file_path: Path) -> Dict:
        """Check all docstrings in a Python file."""
        result = {
            'file': str(file_path),
            'functions': {},
            'classes': {},
            'errors': [],
            'total_issues': 0
        }
        
        try:
            # Load module
            spec = importlib.util.spec_from_file_location("temp_module", file_path)
            if spec is None or spec.loader is None:
                result['errors'].append("Could not load module specification")
                return result
            
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Check all public functions and classes
            for name in dir(module):
                if name.startswith('_'):
                    continue
                
                obj = getattr(module, name)
                full_name = f"{file_path.stem}.{name}"
                
                if inspect.isfunction(obj):
                    issues = self.validate_function_docstring(obj, full_name)
                    if issues:
                        result['functions'][name] = issues
                        result['total_issues'] += len(issues)
                
                elif inspect.isclass(obj):
                    class_issues = self.validate_class_docstring(obj, full_name)
                    method_issues = {}
                    
                    # Check class methods
                    for method_name in dir(obj):
                        if method_name.startswith('__') and method_name not in ['__init__']:
                            continue
                        
                        try:
                            method = getattr(obj, method_name)
                            if inspect.ismethod(method) or inspect.isfunction(method):
                                m_issues = self.validate_method_docstring(method, name, method_name)
                                if m_issues:
                                    method_issues[method_name] = m_issues
                        except Exception:
                            continue
                    
                    if class_issues or method_issues:
                        result['classes'][name] = {
                            'class_issues': class_issues,
                            'method_issues': method_issues
                        }
                        result['total_issues'] += len(class_issues) + sum(len(issues) for issues in method_issues.values())
        
        except Exception as e:
            result['errors'].append(f"Error processing module: {str(e)}")
        
        return result
    
    def check_project(self, project_root: Path) -> Dict:
        """Check all Python files in the project."""
        print("🔍 Finding Python files...")
        
        # Find Python files
        python_files = []
        for py_file in project_root.rglob('*.py'):
            # Skip test files, __pycache__, etc.
            if any(part.startswith('__pycache__') or part.startswith('.') for part in py_file.parts):
                continue
            if 'test' in py_file.name.lower():
                continue
            python_files.append(py_file)
        
        print(f"📄 Found {len(python_files)} Python files to check")
        
        results = {
            'files_checked': 0,
            'total_issues': 0,
            'file_results': {}
        }
        
        for py_file in python_files:
            print(f"  Checking {py_file.relative_to(project_root)}...")
            
            file_result = self.check_module_from_file(py_file)
            results['files_checked'] += 1
            results['total_issues'] += file_result['total_issues']
            
            if file_result['total_issues'] > 0 or file_result['errors']:
                results['file_results'][str(py_file)] = file_result
        
        return results


def main():
    """Main function."""
    print("📚 DiffCT Docstring Completeness Checker")
    print("=" * 50)
    
    # Find project root
    docs_dir = Path(__file__).parent
    project_root = docs_dir.parent
    
    if not project_root.exists():
        print("❌ Could not find project root")
        return 1
    
    # Initialize checker
    checker = DocstringChecker()
    
    # Run checks
    results = checker.check_project(project_root)
    
    # Report results
    print("\n" + "=" * 50)
    print("📊 DOCSTRING CHECK RESULTS")
    print("=" * 50)
    
    print(f"\n📈 Summary:")
    print(f"  • Files checked: {results['files_checked']}")
    print(f"  • Functions checked: {checker.stats['functions_checked']}")
    print(f"  • Classes checked: {checker.stats['classes_checked']}")
    print(f"  • Methods checked: {checker.stats['methods_checked']}")
    print(f"  • Missing docstrings: {checker.stats['missing_docstrings']}")
    print(f"  • Incomplete docstrings: {checker.stats['incomplete_docstrings']}")
    print(f"  • Total issues: {results['total_issues']}")
    
    # Show detailed results
    if results['file_results']:
        print(f"\n📄 Files with issues:")
        
        for file_path, file_result in results['file_results'].items():
            print(f"\n  📁 {Path(file_path).name}")
            
            if file_result['errors']:
                for error in file_result['errors']:
                    print(f"     ❌ Error: {error}")
            
            # Function issues
            for func_name, issues in file_result['functions'].items():
                print(f"     🔧 Function {func_name}:")
                for issue in issues:
                    severity_icon = {'error': '❌', 'warning': '⚠️', 'info': 'ℹ️'}.get(issue['severity'], '•')
                    print(f"        {severity_icon} {issue['message']}")
            
            # Class issues
            for class_name, class_data in file_result['classes'].items():
                print(f"     📦 Class {class_name}:")
                
                for issue in class_data['class_issues']:
                    severity_icon = {'error': '❌', 'warning': '⚠️', 'info': 'ℹ️'}.get(issue['severity'], '•')
                    print(f"        {severity_icon} {issue['message']}")
                
                for method_name, method_issues in class_data['method_issues'].items():
                    print(f"        🔧 Method {method_name}:")
                    for issue in method_issues:
                        severity_icon = {'error': '❌', 'warning': '⚠️', 'info': 'ℹ️'}.get(issue['severity'], '•')
                        print(f"           {severity_icon} {issue['message']}")
    
    # Final status
    error_count = sum(1 for file_result in results['file_results'].values() 
                     for func_issues in file_result['functions'].values() 
                     for issue in func_issues if issue['severity'] == 'error')
    
    error_count += sum(1 for file_result in results['file_results'].values() 
                      for class_data in file_result['classes'].values() 
                      for issue in class_data['class_issues'] if issue['severity'] == 'error')
    
    error_count += sum(1 for file_result in results['file_results'].values() 
                      for class_data in file_result['classes'].values() 
                      for method_issues in class_data['method_issues'].values() 
                      for issue in method_issues if issue['severity'] == 'error')
    
    if error_count == 0:
        print("\n✅ All critical docstring requirements met!")
        if results['total_issues'] > 0:
            print(f"   (Found {results['total_issues']} minor issues that could be improved)")
        return 0
    else:
        print(f"\n❌ Found {error_count} critical docstring issues")
        return 1


if __name__ == "__main__":
    sys.exit(main())