---
type: Documentation
title: Mermaid Diagram Best Practices
description: Guidelines for creating robust Mermaid diagrams in OpenWiki to avoid parsing errors.
tags: [mermaid, diagrams, documentation, best practices]
---

# Mermaid Diagram Best Practices

This document provides best practices for writing Mermaid diagrams in OpenWiki to ensure they render correctly and avoid common parsing errors.

## General Guidelines

*   **Quote node labels**: Always enclose node labels in double quotes (e.g., `A["Node Label"]`) if they contain:
    *   Special characters like parentheses `()`, brackets `[]`, or curly braces `{}`.
    *   Colons `:` (especially in flowcharts).
    *   Reserved Mermaid keywords or syntax.
    *   Any text that might be interpreted as part of Mermaid's syntax.
*   **Avoid Greek letters and symbols**: While Mermaid can sometimes render Unicode characters, it's safer to use their English equivalents (e.g., `alpha` for `α`, `theta` for `θ`, `phi` for `φ`) within node labels to prevent parsing issues across different Mermaid versions or renderers.
*   **Use appropriate node shapes**: Choose the correct node shape syntax for your diagram type. For example, `[]` for rectangular nodes, `()` for rounded nodes, `{}` for rhombus (diamond) nodes in flowcharts, etc. If a label contains complex text, a simpler node shape like a rectangle (`[]`) with a quoted label is often more robust.
*   **Keep labels concise**: Shorter, descriptive labels are generally easier to read and less prone to parsing errors.
*   **Validate diagrams**: If possible, use a Mermaid live editor or a local renderer to validate your diagram syntax before committing it.

By following these guidelines, you can create more robust and error-free Mermaid diagrams for your OpenWiki documentation.