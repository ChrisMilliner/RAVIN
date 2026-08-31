# RAVIN Next.js Prototype

This project converts the original static RAVIN website into a Next.js App Router application.

## Routes

- `/` - Home page
- `/about` - Short user-facing explanation of RAVIN
- `/question` - Policy question form
- `/result` - Prototype loading, answer, and citation state

The question and result flow remains a front-end prototype. The submitted question is stored in the browser's local storage and the result page displays a placeholder response after a short delay. No policy retrieval service or AI backend is connected yet.

## Run locally

Install Node.js 20.9 or newer, then run:

```bash
npm install
npm run dev
```

Open `http://localhost:3000` in a browser.

## Production build

```bash
npm run build
npm start
```
