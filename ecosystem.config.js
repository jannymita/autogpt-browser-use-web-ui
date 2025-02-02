module.exports = {
  apps: [
    {
      name: "python-webserver", // Name of the application
      script: "webui.py", // Your Python script
      interpreter: "python3", // Specify Python interpreter
      interpreter_args: "-m venv .venv && source .venv/bin/activate && python3", // Activate the virtual environment and run the script
      cwd: "./", // Current working directory
      watch: false, // Disable watching for changes (set to true if you want to watch and auto-restart on file changes)
      env: {
        NODE_ENV: "production", // Set environment variable (optional)
      },
      log_file: "./logs/webserver.log", // Optional log file path
    },
  ],
};
