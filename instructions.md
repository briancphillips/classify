# Coding Agent Instructions

## Required Analysis Steps

Before making any code changes, follow these steps carefully and provide your analysis.

### 1. File Indexing
- List all relevant files in scope
- For each file:
  - Current purpose and functionality
  - Key components and features
  - Structure and organization
  - Critical sections that may be affected

### 2. Context Gathering
- Document current implementation details
- Note existing:
  - Design patterns
  - Coding conventions
  - Architecture decisions
  - Dependencies
- Identify:
  - Impact areas
  - Technical constraints
  - Business requirements
  - Performance considerations

### 3. Solution Design
Present 2-3 potential approaches, including:
- Detailed technical approach
- Advantages and disadvantages
- Risk assessment
  - Technical risks
  - Integration challenges
  - Performance implications
- Implementation complexity
- Resource requirements
- Timeline estimation

### 4. Change Management
- Present recommended solution with justification
- **IMPORTANT**: Wait for explicit approval before:
  - Making any code changes
  - Modifying existing functionality
  - Adding new features
  - Refactoring code
  - Updating dependencies

Exception: Changes can proceed without confirmation ONLY for specifically discussed tasks within the current conversation.

## Usage Instructions
1. Reference this file at the start of each new task
2. Follow each section methodically
3. Present findings clearly
4. Wait for approval before implementation
5. If additional changes are needed beyond the approved scope, seek new confirmation

### 5. Commit Process
- Create a descriptive branch name using format: `feature/task-description` or `fix/issue-description`
- Stage changes with clear grouping
- Create commits with:
  - Subject line: Brief (50 chars or less) summary in imperative mood
  - Body: Detailed explanation of:
    - What was changed
    - Why the change was made
    - Any notable implementation decisions
  - Reference any relevant issue numbers
- Example commit message:
  ```
  Add user authentication middleware

  - Implement JWT token verification
  - Add rate limiting for login attempts
  - Create user session management
  
  This change improves security by adding proper
  authentication checks before protected routes.
  
  Closes #123
  ```
- For multiple commits, provide a summary of all changes
- Request commit review before pushing

Always respond with:
1. Complete analysis following these steps
2. Clear recommendation
3. Request for approval before proceeding