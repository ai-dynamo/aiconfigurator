# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import re
import tempfile


def safe_mkdir(target_path: str, exist_ok: bool = True) -> Path:
    """
    Safely create a directory with path validation, sanitization, and security checks.
    
    This function validates the parent directory for security, sanitizes the target
    directory name, and creates the directory using pathlib.
    
    Args:
        target_path: The target directory path to create
        exist_ok: If True, don't raise an exception if the directory already exists
        
    Returns:
        Path: The resolved absolute path of the created directory
        
    Raises:
        ValueError: If the path is invalid or outside allowed directories
        OSError: If directory creation fails
    """
    def _sanitize_path_component(component: str) -> str:
        """
        Sanitize a path component (closure function).
        """
        if not component:
            return "unknown"
        
        # Replace dangerous characters with underscores
        sanitized = re.sub(r'[^\w\-_.]', '_', str(component))
        
        # Remove leading/trailing dots and spaces
        sanitized = sanitized.strip('. ')
        
        # Ensure it's not empty after sanitization
        if not sanitized:
            return "unknown"
        
        # Limit length to prevent extremely long filenames
        return sanitized[:100]
    
    if not target_path:
        raise ValueError("Target path cannot be empty")
    
    try:
        # Parse the target path
        target = Path(target_path)
        
        # Get parent directory and target directory name
        if target.is_absolute():
            # For absolute paths, validate the entire path
            parent_dir = target.parent
            dir_name = target.name
        else:
            # For relative paths, validate from current directory
            parent_dir = Path.cwd()
            # Split the relative path and sanitize each component
            parts = target.parts
            sanitized_parts = [_sanitize_path_component(part) for part in parts]
            
            # Build the final path
            final_target = parent_dir
            for part in sanitized_parts:
                final_target = final_target / part
            
            return safe_mkdir(str(final_target), exist_ok)
        
        # Validate parent directory security
        resolved_parent = parent_dir.resolve()
        
        # Security check: ensure no null bytes
        if '\x00' in str(resolved_parent):
            raise ValueError("Path contains null byte")
        
        # Check if the parent path is within allowed locations
        current_dir = Path.cwd().resolve()
        allowed_prefixes = [
            current_dir,
            Path.home(),
            Path('/tmp'),
            Path('/workspace'),
            Path('/var/tmp'),
            Path(tempfile.gettempdir()).resolve(),
        ]
        
        # Verify the parent path is under an allowed prefix
        is_allowed = any(
            resolved_parent == prefix or resolved_parent.is_relative_to(prefix)
            for prefix in allowed_prefixes
        )
        
        if not is_allowed:
            raise ValueError(f"Path is outside allowed locations: {resolved_parent}")
        
        # Sanitize the target directory name and create final path
        sanitized_name = _sanitize_path_component(dir_name)
        final_path = resolved_parent / sanitized_name
        
        # Create the directory using pathlib
        final_path.mkdir(parents=True, exist_ok=exist_ok)
        
        return final_path
        
    except (OSError, ValueError) as e:
        if isinstance(e, ValueError):
            raise
        raise ValueError(f"Failed to create directory: {e}")