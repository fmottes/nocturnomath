# Browser Automation Policy

Do not invoke the `browser_subagent` or any browser automation tools. Headless browser automation is disabled in this environment pending an official upstream fix from Antigravity/Playwright.

For all frontend and web application testing, use:
- Backend integration tests via `curl`, `urllib`, or `requests`/`httpx`
- Direct WebSocket and REST API verification scripts
- Manual verification in the user's desktop browser
