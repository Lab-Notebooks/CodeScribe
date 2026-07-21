# Copyright (c) 2026 UChicago Argonne LLC
# SPDX-License-Identifier: Apache-2.0
# Full license and notices: see LICENSE and NOTICE in the repo root.

import re
import os
import toml
import yaml


from pathlib import Path
from textwrap import wrap
from typing import List, Dict, Set, Any

from codescribe import lib

from ._logging import write_archive_toml

__all__ = [
    "load_chat_template",
    "format_seed_prompt",
    "format_text_block",
    "extract_fortran_info",
    "create_scribe_yaml",
    "load_scribe_yaml",
    "create_file_indexes",
    "filter_file_indexes",
    "isolate_scalar_functions",
    "query_construct",
    "extract_fortran_meta",
    "annotate_fortran_file",
    "create_src_mapping",
]


def load_chat_template(filepath: Path, *, return_meta: bool = False):
    """Load and validate a chat prompt file.

    Chat format:
      - [[chat.user]] / [[chat.assistant]]
        content = ''' ... '''

    Legacy format (still supported):
      - [[chat]]
        role = "user" / "assistant"
        content = ''' ... '''

    Optional tool policy:
      [tools]
      bash = ["python3.8", "rg"]

    Validation rules:
      - Conversation must start with a 'user' message
      - Roles must alternate strictly: user-assistant-user-assistant ...
      - Conversation must end with a 'user'
      - No two consecutive roles may be identical
      - Multi-line content **must** use triple single quotes (''' ... ''')

    Returns:
      - return_meta=False (default): chat_template
      - return_meta=True: (chat_template, meta)
        meta currently includes: {"tools": {"bash": [..]}}
    """
    path = Path(filepath)
    if path.suffix != ".toml":
        raise ValueError(
            f"Unsupported file extension '{path.suffix}'. Expected '.toml'."
        )

    raw_text = path.read_text()
    data = toml.loads(raw_text)

    # Optional tool policy (backwards-compatible: absence is fine)
    meta: Dict[str, Any] = {}
    tools_cfg = data.get("tools") if isinstance(data, dict) else None
    if tools_cfg is not None:
        if not isinstance(tools_cfg, dict):
            raise TypeError(f"[tools] must be a table in {filepath}")
        bash_tools = tools_cfg.get("bash", [])
        if bash_tools is None:
            bash_tools = []
        if not isinstance(bash_tools, list) or not all(
            isinstance(x, str) for x in bash_tools
        ):
            raise TypeError(f"[tools].bash must be a list of strings in {filepath}")

        # Name-only: no args, no paths.
        name_re = re.compile(r"^[A-Za-z0-9_.+-]+$")
        for x in bash_tools:
            if not name_re.match(x):
                raise ValueError(
                    f"Invalid bash tool name {x!r} in {filepath}. "
                    "Use a bare command name only (e.g. 'python3.8'), no args or paths."
                )

        meta = {"tools": {"bash": bash_tools}}

    # --- Enforce triple single quotes for multi-line content ---
    bad_quotes = re.findall(r'content\s*=\s*"""', raw_text)
    if bad_quotes:
        raise ValueError(
            f"Invalid TOML quoting in {filepath}: "
            f"Use triple single quotes ('''...''') for multi-line content blocks "
            f'instead of triple double quotes ("""...""").'
        )

    if "chat" not in data:
        raise KeyError(f"'chat' key not found in TOML file: {filepath}")

    chat_section = data["chat"]
    chat_template = []
    valid_roles = {"user", "assistant"}

    # --- Case 1: [[chat]] array of tables ---
    if isinstance(chat_section, list):
        for entry in chat_section:
            if (
                not isinstance(entry, dict)
                or "role" not in entry
                or "content" not in entry
            ):
                raise KeyError(f"Invalid [[chat]] entry in {filepath}")
            role = entry["role"]
            if role not in valid_roles:
                raise ValueError(f"Invalid role '{role}' in {filepath}")
            chat_template.append({"role": role, "content": entry["content"]})

    # --- Case 2: [[chat.user]] / [[chat.assistant]] arrays of tables ---
    elif isinstance(chat_section, dict):
        pattern = re.compile(r"\[\[chat\.(user|assistant)\]\]")
        declared_roles = [m.group(1) for m in pattern.finditer(raw_text)]
        counters = {role: 0 for role in valid_roles}

        for role in declared_roles:
            if role not in chat_section:
                raise KeyError(
                    f"[[chat.{role}]] section declared but not found in {filepath}"
                )
            entries = chat_section[role]
            idx = counters[role]
            if idx >= len(entries):
                raise IndexError(
                    f"More [[chat.{role}]] headers than parsed entries for {role} in {filepath}"
                )
            entry = entries[idx]
            if "content" not in entry:
                raise KeyError(f"[[chat.{role}]] entry missing 'content' in {filepath}")
            chat_template.append({"role": role, "content": entry["content"]})
            counters[role] += 1
    else:
        raise TypeError(f"Unexpected 'chat' structure in {filepath}")

    # --- Conversation alternation validation ---
    if not chat_template:
        raise ValueError("Chat template is empty.")

    roles = [entry["role"] for entry in chat_template]
    if roles[0] != "user":
        raise ValueError("Conversation must start with a 'user' message.")
    if roles[-1] == "assistant":
        raise ValueError("Conversation must end with a 'user' message.")
    for i in range(1, len(roles)):
        if roles[i] == roles[i - 1]:
            raise ValueError(
                f"Invalid role order at positions {i} and {i + 1}: "
                f"two consecutive '{roles[i]}' entries found."
            )
    return (chat_template, meta) if return_meta else chat_template


def format_seed_prompt(filepath: Path, chat_entries: List[Dict[str, str]] = []) -> None:
    """
    Format TOML chat file in place using format_text_block().
    - Preserves all comments, headers, and blank lines exactly.
    - Uses chat_entries (from load_chat_template) as the source of truth.
    - Replaces each content block sequentially in the raw file.
    """

    path = Path(filepath)
    raw_text = path.read_text()

    if not chat_entries:
        chat_entries = load_chat_template(filepath)

    # Match any TOML content block: content = ''' ... ''' OR content = """ ... """
    pattern = re.compile(
        r"content\s*=\s*(?P<quote>'''|\"|\')([\s\S]*?)(?P=quote)", re.MULTILINE
    )

    parts = []
    last_end = 0
    entry_idx = 0

    for match in pattern.finditer(raw_text):
        start, end = match.span()
        quote = match.group("quote")
        content_inside = match.group(2)

        # Keep comments or non-chat lines before this unchanged
        parts.append(raw_text[last_end:start])

        if entry_idx < len(chat_entries):
            entry = chat_entries[entry_idx]
            formatted = format_text_block(entry["content"])
            new_block = f"content = '''\n{formatted}\n'''"
            parts.append(new_block)
            entry_idx += 1
        else:
            # Fallback: leave unrecognized content untouched
            parts.append(match.group(0))

        last_end = end

    # Preserve everything after last content block
    parts.append(raw_text[last_end:])
    formatted_text = "".join(parts)

    # Normalize spacing (optional cleanup)
    formatted_text = re.sub(r"\n{3,}", "\n\n", formatted_text)

    # Ensure no accidental regex backreferences
    formatted_text = formatted_text.replace("\\1", "").replace("\\3", "")

    path.write_text(formatted_text)


def format_text_block(text: str, width: int = 100, indent_step: int = 3) -> str:
    """
    Cleanly format a text block (like Markdown or TOML content).

    - Wraps lines to 'width' without breaking code fences.
    - Removes trailing whitespace.
    - Normalizes indentation levels to multiples of indent_step.
    - Collapses extra blank lines.

    Example:
        formatted = format_text_block(raw_text, width=100, indent_step=2)
    """
    lines = text.splitlines()
    out_lines = []
    in_code_block = False

    for line in lines:
        stripped = line.rstrip()

        stripped = stripped.replace("\t", " " * 8)
        indent = len(stripped) - len(stripped.lstrip())

        if stripped.startswith("-"):
            stripped = "-" + " " * 2 + stripped.split("-", 1)[1].lstrip()

        if indent > 0:
            indent = max(indent + 1, 3) // indent_step * indent_step
        indent_str = " " * indent

        # reflow line if too long
        if len(stripped) > width:
            wrapped_lines = wrap(stripped.lstrip(), width=width - indent)
            out_lines.extend(indent_str + w for w in wrapped_lines)
        else:
            out_lines.append(indent_str + stripped.lstrip())

    # collapse multiple blank lines
    formatted = []
    last_blank = False
    for l in out_lines:
        if not l.strip():
            if not last_blank:
                formatted.append("")
            last_blank = True
        else:
            formatted.append(l)
            last_blank = False

    return "\n".join(formatted).rstrip() + "\n"


def extract_fortran_info(filepath: Path) -> Dict[str, List[str]]:
    """Extracts module and subroutine/function names from a Fortran file."""
    info = {"modules": [], "subroutines": [], "functions": []}

    with open(filepath, "r") as file:
        for line in file:
            line = line.strip()
            line = line.lower()
            # Check for module declaration
            if line.startswith("module "):
                info["modules"].append(line.split()[1])  # Capture module name
            # Check for subroutine declaration
            elif line.startswith("subroutine "):
                # Extract the subroutine name (first word after "subroutine")
                match = re.match(r"subroutine\s+(\w+)", line)
                if match:
                    info["subroutines"].append(
                        match.group(1)
                    )  # Capture subroutine name
            # Check for function declaration
            elif line.startswith("function "):
                # Extract the function name (first word after "function")
                match = re.match(r"function\s+(\w+)", line)
                if match:
                    info["functions"].append(match.group(1))  # Capture function name

    return info


def create_scribe_yaml(root_directory: Path) -> None:
    """Traverses the directory and creates scribe.yaml files for Fortran files."""
    for dirpath, _, filenames in os.walk(root_directory):
        scribe_data = {
            "root": str(root_directory),
            "directory": dirpath.replace(str(root_directory) + os.sep, ""),
            "files": {},
        }

        for filename in filenames:
            if filename.endswith((".f", ".f90", ".F", ".F90")):
                filepath = os.path.join(dirpath, filename)
                fortran_info = extract_fortran_info(filepath)

                # Add extracted information to the scribe_data
                scribe_data["files"][filename] = fortran_info

        # Only write to scribe.yaml if there are Fortran files in the directory
        # if scribe_data["files"]:
        yaml_path = os.path.join(dirpath, "scribe.yaml")
        with open(yaml_path, "w") as yaml_file:
            yaml.dump(scribe_data, yaml_file, default_flow_style=False)


def load_scribe_yaml(file_path: Path) -> Dict[str, str]:
    """Load the content of a scribe.yaml file."""
    with open(file_path, "r") as yaml_file:
        return yaml.safe_load(yaml_file)


def create_file_indexes() -> Dict[str, str]:
    """
    Create a combined index for files, subroutines, functions,
    and modules from all scribe.yaml files in the directory tree.
    """

    # Start with the current working directory
    cwd = os.getcwd()

    # Try to find scribe.yaml: first check cwd, then search downward
    yaml_path = os.path.join(cwd, "scribe.yaml")
    if not os.path.exists(yaml_path):
        # Search downward until we find the first scribe.yaml
        yaml_path = None
        for dirpath, dirnames, filenames in os.walk(cwd):
            # Sort dirnames for deterministic traversal order
            dirnames.sort()
            if "scribe.yaml" in filenames:
                yaml_path = os.path.join(dirpath, "scribe.yaml")
                break

        # No scribe.yaml found anywhere under cwd; skip gracefully
        if yaml_path is None:
            return {}

    # Load the anchor scribe.yaml
    try:
        scribe_data = load_scribe_yaml(Path(yaml_path))
    except Exception:
        return {}

    # Get the root directory from the scribe.yaml file
    root_directory = scribe_data.get("root", None)
    if not root_directory:
        return {}

    file_index = {}

    # Traverse the directory tree starting from the root directory
    for dirpath, _, filenames in os.walk(root_directory):
        if "scribe.yaml" not in filenames:
            continue

        filepath = os.path.join(dirpath, "scribe.yaml")
        try:
            scribe_data = load_scribe_yaml(Path(filepath)) or {}
        except Exception:
            continue

        # Update combined index with data from the current scribe.yaml
        for file, info in (scribe_data.get("files") or {}).items():
            file_path = os.path.join(dirpath, file)  # Full file path

            # Extract modules, subroutines, and functions
            modules = info.get("modules", [])
            subroutines = info.get("subroutines", [])
            functions = info.get("functions", [])

            # Add modules to combined index
            for mod in modules:
                file_index[mod] = file_path
            # Add subroutines to combined index
            for sub in subroutines:
                file_index[sub] = file_path
            # Add functions to combined index
            for func in functions:
                file_index[func] = file_path

    return file_index


def filter_file_indexes(
    sfile: Path, file_index: Dict[str, str], function_calls: Set[str] = []
) -> Dict[str, Path]:
    """
    Extract modules, subroutines, and variable declarations used in the given
    Fortran source file, and return a subset of the file_index that corresponds to these.
    """
    used_modules = set()
    used_subroutines = set()

    # Open and read the source file
    try:
        with open(sfile, "r") as source:
            for line in source:
                stripped_line = line.strip()

                # Check for 'use' statement to capture modules
                module_match = re.match(
                    r"^\s*use\s+(\w+)", stripped_line, re.IGNORECASE
                )
                if module_match:
                    used_modules.add(module_match.group(1).lower())

                # Check for 'call' statement to capture subroutines
                subroutine_match = re.match(
                    r"^\s*call\s+(\w+)", stripped_line, re.IGNORECASE
                )
                if subroutine_match:
                    used_subroutines.add(subroutine_match.group(1).lower())

    except FileNotFoundError:
        print(f"Error: The file '{sfile}' was not found.")
        return {}
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return {}

    # Filter the file_index to return only the modules, subroutines, and
    # variables used in this file
    filtered_file_index = {
        name: path
        for name, path in file_index.items()
        if name.lower() in used_modules
        or name.lower() in used_subroutines
        or name.lower() in [sfunc.lower() for sfunc in function_calls]
        and path != os.path.abspath(sfile)
    }

    return filtered_file_index


def isolate_scalar_functions(sfile: Path) -> List[List[str]]:
    """
    Isolate variables that are declared as scalars but used as functions
    in the given Fortran source file.
    """
    scalar_variables = set()
    function_calls = set()
    defined_functions = set()

    # Open and read the source file
    with open(sfile, "r") as source:
        for line in source:
            stripped_line = line.strip()

            # Check for function definitions
            func_decl_match = re.match(
                r"^\s*function\s+(\w+)", stripped_line, re.IGNORECASE
            )
            if func_decl_match:
                defined_functions.add(func_decl_match.group(1))

            # Check for scalar variable declarations
            var_decl_match = re.match(
                r"^\s*(integer|real|double|complex\(dp\)|bool|character)\s*::\s*([^;]*)",
                stripped_line,
                re.IGNORECASE,
            )
            if var_decl_match:
                # Extract variable names
                variables_part = var_decl_match.group(2)
                variable_names = [
                    var.strip().split("(")[0] for var in variables_part.split(",")
                ]
                scalar_variables.update(name for name in variable_names)

            # Check for function calls
            function_call_match = re.findall(r"\b(\w+)\s*\(", stripped_line)
            if function_call_match:
                function_calls.update(name for name in function_call_match)

    # Identify scalar variables that are used as functions, ignoring defined functions
    scalar_used_as_functions = (
        scalar_variables.intersection(function_calls) - defined_functions
    )

    return scalar_used_as_functions, function_calls


def query_construct(name: str, file_index: Dict[str, str]) -> List[str]:
    """Query the file path of a module, subroutine, or function."""

    # Find all matches for the given name
    matches = [
        file_path for construct, file_path in file_index.items() if name == construct
    ]

    return matches if matches else None


def extract_fortran_meta(sfile: Path) -> Dict[str, Any]:
    """
    Extract function, subroutine, module names, variables, and argument
    lists from Fortran source.
    """

    meta_info = []
    current_construct = None
    variables_declared = []
    argument_list = []

    with open(sfile, "r") as source:
        for line in source.readlines():
            stripped_line = line.strip()

            # Check for subroutine, module, or function start
            construct_match = re.match(
                r"^\s*(subroutine|function|module)\s+(\w+)",
                stripped_line,
                re.IGNORECASE,
            )
            if construct_match:
                # If we were already inside a construct, save the previous one
                if current_construct:
                    meta_info.append(
                        {
                            "name": current_construct["name"],
                            "type": current_construct["type"],
                            "variables_declared": variables_declared,
                            "argument_list": argument_list,
                        }
                    )

                # Start a new construct
                current_construct = {
                    "name": construct_match.group(2),
                    "type": construct_match.group(1).lower(),
                }
                variables_declared = []  # Reset variables declared
                argument_list = []  # Reset argument list

                # If it's a function or subroutine, look for the argument list
                if current_construct["type"] in ["function", "subroutine"]:
                    args_match = re.search(r"\((.*?)\)", stripped_line)
                    if args_match:
                        arguments = [
                            arg.strip() for arg in args_match.group(1).split(",")
                        ]
                        argument_list.extend(arguments)

            # Extract variable declarations (e.g., integer, real, character, etc.)
            var_match = re.match(
                r"^\s*(integer|real|double\s*precision|character|logical)\s+([\w\s,]*)",
                stripped_line,
                re.IGNORECASE,
            )
            if var_match:
                variables = [var.strip() for var in var_match.group(2).split(",")]
                variables_declared.extend(variables)

        # Save the last construct if it exists
        if current_construct:
            meta_info.append(
                {
                    "name": current_construct["name"],
                    "type": current_construct["type"],
                    "variables_declared": variables_declared,
                    "argument_list": argument_list,
                }
            )

    return meta_info


def annotate_fortran_file(sfile: Path, file_index: Dict[str, str]) -> str:
    """
    Annotates a Fortran file, converts types to C++ equivalents,
    replaces use statements inline with namespaces, and adds headers.
    """

    scalar_functions, function_calls = lib.isolate_scalar_functions(sfile)

    filtered_file_index = {}
    if file_index:
        filtered_file_index.update(
            lib.filter_file_indexes(sfile, file_index, function_calls)
        )

    scribe_filename = os.path.splitext(sfile)[0] + ".scribe"

    if os.path.isfile(scribe_filename):
        return f"Skipping! File exists {scribe_filename}..."

    header_includes = set(
        ("#include <cmath>", "#include <complex>")
    )  # Keep track of headers to avoid duplicates
    content_lines = []  # Store lines of modified content
    prompt_lines = []

    prompt_lines.append(
        'scribe-prompt: Write corressponding extern "C" with _wrapper added to the name. '
        + "Refer to the template for treating Farray and scalars"
    )
    prompt_lines.append(
        "scribe-prompt: When variables are used as function. They should be treated as "
        + "external or statement functions. "
        + "External functions are available in header files"
    )
    prompt_lines.append(
        "scribe-prompt: Statement functions should be converted to equivalent lambda "
        + "functions in C++. Include [&] in capture clause to use variables by reference"
    )

    for construct in function_calls:
        if construct.lower() in filtered_file_index.keys():
            prompt_lines.append(f"scribe-prompt: {construct} is an external function")

    for construct in scalar_functions:
        prompt_lines.append(
            f"scribe-prompt: {construct} is an array or statement function"
        )

    with open(sfile, "r") as source:
        source_code = source.readlines()

        for line in source_code:
            stripped_line = line.strip()

            # Skip lines that are comments starting with 'c', '!!', or '!'
            if stripped_line.lower().startswith(("c", "!!", "!")) and (
                not stripped_line.lower().startswith(("complex"))
                and not stripped_line.lower().startswith(("call"))
            ):
                continue

            # Replace 'use <module_name>' with '#include' and 'using namespace'
            use_match = re.match(r"\buse\s+(\w+)", stripped_line, flags=re.IGNORECASE)
            if use_match:
                module_name = use_match.group(1)

                # Add header include globally but only once
                header_includes.add(f"#include <{module_name}.hpp>")

                # Replace 'use' with 'using namespace' inline
                content_lines.append(f"using namespace {module_name};\n")
                continue

            # Remove implict none statement
            line = re.sub(r"implicit none", "", line)

            # Handle variable declarations and conversions
            line = re.sub(r"\binteger\b\s*", "int", line, flags=re.IGNORECASE)
            line = re.sub(
                r"\breal\s*(\(\s*kind\s*=\s*\w+\s*\)|\(\s*\w+\s*\)|)?\s*",
                "double",
                line,
                flags=re.IGNORECASE,
            )
            line = re.sub(
                r"\bcomplex\s*\(\s*dp\s*\)\s*",
                "complex<double> ",
                line,
                flags=re.IGNORECASE,
            )
            line = re.sub(
                r"\bcomplex\s*\(\s*integer\s*\)\s*",
                "complex<int> ",
                line,
                flags=re.IGNORECASE,
            )
            line = re.sub(
                r"\bcomplex\s*\(\s*logical\s*\)\s*",
                "complex<bool> ",
                line,
                flags=re.IGNORECASE,
            )
            line = re.sub(r"(?<!std::)\s*::", "", line)

            # Handle complex types in variable declarations, ensuring dimensionality is handled
            line = re.sub(
                r"\bcomplex<([^>]+)>\s*(\w+)\s*\((.*?)\)\s*",
                r"FArray<std::complex<\1>> \2(\3)",
                line,
            )

            line = re.sub(
                r"\b(real|double|int|bool|complex<[^>]+>)\s*,?\s*dimension\s*\((.*?)\)\s*(\w+)\s*;",
                r"FArray<\1> \3(\2)",
                line,
                flags=re.IGNORECASE,
            )

            line = re.sub(
                r"\b(real|double|int|bool|complex<[^>]+>)\s*(\w+)\s*\((.*?)\)\s*;",
                r"FArray<\1> \2(\3)",
                line,
                flags=re.IGNORECASE,
            )

            # Treat line continuation characters. Replace them with equivalent syntax in C++
            line = re.sub(r"^\s*&", r"\\", line)
            line = re.sub(r"\s*&\s*$", r" \\", line)

            # Substitution x**y with pow(x,y)
            line = re.sub(r"(\w+)\s*\*\*\s*(\d+)", r"pow(\1,\2)", line)

            # Add a semicolon at the end of variable declarations
            # if re.match(r"^(int|double|complex<[^>]+>)\s", line.strip()):
            #    line = line.strip() + ";"

            # Append the modified line to content_lines
            content_lines.append(line.strip() + "\n")

    # Write the output to the .scribe file
    with open(scribe_filename, "w") as scribe_file:
        scribe_file.write("\n".join(prompt_lines))
        scribe_file.write("\n\n")

        # First, write all the includes at the top
        if header_includes:
            scribe_file.write("\n".join(sorted(header_includes)) + "\n\n")

        # Then, write the rest of the modified content
        scribe_file.writelines(content_lines)

    return f"Generated draft file for LLM consumption {scribe_filename}"


def create_src_mapping(filelist: List[Path]) -> List[str]:
    """
    Build directory tree from neucol source code.

    Arguments
    ---------
    dest : String value of destination directory
    """
    fsource = []
    csource = []
    cheader = []
    finterface = []
    cdraft = []
    promptfile = []

    for sfile in filelist:
        fsource.append(sfile)
        csource.append(os.path.splitext(sfile)[0] + ".cpp")
        cheader.append(os.path.splitext(sfile)[0] + ".hpp")
        finterface.append(os.path.splitext(sfile)[0] + "_fi.F90")
        cdraft.append(os.path.splitext(sfile)[0] + ".scribe")
        promptfile.append(os.path.splitext(sfile)[0] + ".json")

    return fsource, csource, finterface, cdraft, promptfile, cheader
