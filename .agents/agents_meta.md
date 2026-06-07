# agents.md

This file provides guidance to Claude Code and other AI assistants when working with projects in this workspace.

## Repository Structure

This directory (`/Users/isaacsim/projects/.agents`) contains global metadata and configuration for AI assistants across all projects in the `projects` folder.

- **Parent folder**: `/Users/isaacsim/projects/`
- **Purpose**: Store common metadata and instructions that apply to all projects
- **Project structure**: Each project subfolder in `projects/` will have its own `.agents/` directory for project-specific AI metadata

## Automatic Project Setup Instructions

When working in any project folder under `/Users/isaacsim/projects/[project-name]/`:

1. **Check for `.agents/` directory**: If the project folder doesn't have a `.agents/` subdirectory, create it automatically.

2. **Check for `.agents/agents_meta.md` file**: If `.agents/agents_meta.md` doesn't exist in the project folder, create it by copying the entire contents of this file (`/Users/isaacsim/projects/.agents/agents.md`).

3. **Check for `.agents/agents.md` file**: If `.agents/agents.md` doesn't exist in the project folder, create it with the following base structure:
   ```markdown
   # Project AI Instructions

   This file contains project-specific instructions for AI assistants working on this project.

   ## Base Instructions

   This project follows the global AI metadata defined in `.agents/agents_meta.md`.

   ## Project-Specific Instructions

   [Add custom instructions here for this specific project]

   ## Auto-managed Files

   The following files are automatically read and updated based on this instruction file:
   - `.agents/agents_meta.md` - Project metadata and architecture
   - `.agents/plan.md` - Task planning and progress tracking
   - `.agents/consult.md` - Consultation history and decisions
   ```

4. **Check for `.agents/plan.md` file**: If it doesn't exist, create it with the following structure:
   - **진행 완료 (Completed)**
   - **진행 중 (In Progress)**
   - **진행 예정 (To Do)**

   Each task entry must start with `[대주제]` format. If unsure about the appropriate 대주제 (main category), ask the user first. The 대주제 can be based on:
   - Task type (e.g., [기능개발], [버그수정], [리팩토링])
   - Domain/area (e.g., [백엔드], [프론트엔드], [데이터베이스])

   Once determined, save the 대주제 guideline in the project's `.agents/agents_meta.md` for consistency.

5. **Check for `.agents/consult.md` file**: If it doesn't exist, create it. This file records consultation summaries between the user and AI assistant:
   - Record key questions from the user
   - Record summary of AI responses
   - Include context from small to large project decisions (e.g., database selection, architecture choices, testing strategies)
   - Format: User's question intent + AI's answer summary

6. **File structure**:
   ```
   /Users/isaacsim/projects/
   ├── .agents/
   │   └── agents.md (this file - global instructions and template)
   └── [project-name]/
       └── .agents/
           ├── agents.md (project-specific instructions, reads agents_meta.md + custom instructions)
           ├── agents_meta.md (project metadata, copied from global agents.md)
           ├── plan.md (task planning and progress tracking)
           └── consult.md (consultation history and decisions)
   ```

## Critical Workflow for Project Sessions

**IMPORTANT**: When starting ANY work session in a project folder (`/Users/isaacsim/projects/[project-name]/`):

1. **ALWAYS read `[project-name]/.agents/agents.md` FIRST**
   - This is the master instruction file for the project
   - It contains references to agents_meta.md and project-specific instructions
   - All subsequent actions are guided by this file

2. **Automatically read the following files** (as referenced in the project's agents.md):
   - `.agents/agents_meta.md` - For project metadata, architecture, and conventions
   - `.agents/plan.md` - For current task status and planning
   - `.agents/consult.md` - For historical decisions and consultations

3. **During the session**:
   - Update `.agents/plan.md` as tasks progress
   - Add consultation summaries to `.agents/consult.md` when significant discussions occur
   - **NEVER automatically update `.agents/agents_meta.md`** - This file is read-only unless user explicitly requests changes

4. **File creation order** (if files don't exist):
   - Create `.agents/` directory
   - Create `.agents/agents_meta.md` (copy from `/Users/isaacsim/projects/.agents/agents.md`)
   - Create `.agents/agents.md` (with base template above)
   - Create `.agents/plan.md` (with structure above)
   - Create `.agents/consult.md` (empty or with header)

5. **Initial Project Analysis** (when `.agents/` folder is first created):
   - **First, check if the project folder has meaningful content**:
     - If the project folder is empty or only contains `.agents/` folder, SKIP analysis
     - Only proceed with analysis if there are actual project files (source code, config files, etc.)

   - **If meaningful content exists**, read and analyze the project folder structure and files to understand:
     - Project purpose and functionality
     - Technology stack and dependencies
     - Architecture and code structure
     - Key features and components

   - **DO NOT immediately save** the analyzed content
   - Present the analysis to the user with the following message:
     ```
     프로젝트를 분석한 결과입니다:

     [분석 내용 요약]

     이 내용을 `.agents/agents.md` 파일에 정리하여 업데이트할까요?
     - 그대로 진행
     - 수정하여 진행
     ```
   - **Wait for user confirmation** before updating `.agents/agents.md`
   - If user approves or provides modifications, update `.agents/agents.md` accordingly

## AGENTS_META.md Management Rules

**CRITICAL**: The `.agents/agents_meta.md` file in project folders is **READ-ONLY** by default.

1. **Never automatically modify** `[project-name]/.agents/agents_meta.md` during normal sessions
2. **Only modify when user explicitly requests** changes to agents_meta.md
3. **When user explicitly requests to modify** `[project-name]/.agents/agents_meta.md`:
   - Make the requested changes to `[project-name]/.agents/agents_meta.md`
   - **Immediately copy the entire contents** of `[project-name]/.agents/agents_meta.md` to `/Users/isaacsim/projects/.agents/agents.md`
   - This keeps the global template synchronized with project-level customizations

## Session Summary Command

When the user says **"지금까지 내역 정리"** (summarize progress so far):

1. Update all relevant `.md` files in the project's `.agents/` directory:
   - `agents.md`: Add any new project-specific instructions if needed
   - `plan.md`: Move completed tasks to "진행 완료", update "진행 중" and "진행 예정" sections
   - `consult.md`: Add summaries of any consultations that occurred in this session
   - Any other `.md` files the user has requested to create
   - **DO NOT update `agents_meta.md`** unless explicitly requested

2. Include timestamps (연월일시) where appropriate, especially for:
   - Task completion dates in `plan.md`
   - Consultation dates in `consult.md`

3. Use format: `YYYY-MM-DD HH:mm` or `YYYY-MM-DD` as appropriate

4. **Clean up temporary files**:
   - Delete `.claude/` folder if it exists (Claude Code CLI auto-generated settings)
   - Remove any cache files (`__pycache__/`, `*.pyc`, `*.pyo`)
   - Remove system files (`.DS_Store`, `Thumbs.db`)
   - Remove log files if not needed (`*.log`)
   - Execute: `rm -rf .claude && find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null; find . -name ".DS_Store" -delete 2>/dev/null`

## Global Metadata Template

The content below serves as the base template for all project-specific `agents_meta.md` files:

---

# Project Metadata

**Project Name**: [To be filled]
**Technology Stack**: [To be filled]
**Purpose**: [To be filled]
**Last Updated**: [To be filled]

## Development Commands

### Build
[Add build commands here]

### Test
[Add test commands here]

### Lint
[Add lint commands here]

### Run
[Add run commands here]

## Architecture Overview

[Add high-level architecture description here]

## Key Conventions

[Add project-specific conventions and patterns here]

## Task Category Guidelines (대주제 가이드라인)

[Define how to categorize tasks - by type or by domain. Examples:
- By type: [기능개발], [버그수정], [리팩토링], [문서화]
- By domain: [백엔드], [프론트엔드], [데이터베이스], [인프라]]

## Notes for AI Assistants

- Follow the established project structure
- Maintain consistency with existing code patterns
- Update this file when significant architectural changes are made
- Use the defined 대주제 categories when updating plan.md
