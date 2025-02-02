module.exports = {
  apps: [
    {
      name: "autogpt-browser-use-web-ui", // Name of the application
      script: "source .venv/bin/activate && python webui.py", // Your Python script
      cwd: "./", // Current working directory
      watch: false, // Disable watching for changes (set to true if you want to watch and auto-restart on file changes)
      env: {
        NODE_ENV: "production", // Set environment variable (optional)
      },
      log_file: "./logs/webserver.log", // Optional log file path
    },
  ],
};
