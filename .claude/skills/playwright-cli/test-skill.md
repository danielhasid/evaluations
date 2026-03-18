The user wants to test the TodoMVC app with playwright-cli in headed mode, taking screenshots for all scenarios (both successful and failing).

Let me plan the test scenarios for TodoMVC:

Successful scenarios:

Add a todo
Complete a todo (check it off)
Delete a todo
Filter by Active
Filter by Completed
Clear completed todos
Edit a todo (double-click)
Mark all as complete
Failing scenarios (things that should fail or edge cases):

Add empty todo (should not add)

Add whitespace-only todo

Try to edit a todo with empty text

Now I'll work through each scenario methodically, launching the app in headed mode and capturing screenshots at each step to document the behavior.

Plan
I'll test these scenarios on TodoMVC in headed mode:

Happy path: Add todo, Complete todo, Delete todo, Edit todo, Filter (Active/Completed), Clear completed, Mark all complete

Edge cases (expected failures): Empty todo submission, Whitespace-only todo