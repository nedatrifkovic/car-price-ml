# Use the official slim Python image as the base
FROM python:3.11-slim

# Do not write .pyc files and run Python in unbuffered mode (helpful for logs)
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Pin Poetry version and configure Poetry to not create virtualenvs (we want system env)
ENV POETRY_VERSION=1.5.1
ENV POETRY_VIRTUALENVS_CREATE=false
ENV POETRY_NO_INTERACTION=1

# Set the working directory inside the container
WORKDIR /app


# Install system packages required to build some Python packages (g++ may be needed for some wheels)
# Clean apt lists afterward to keep image small.
RUN apt-get update \
  && apt-get install -y --no-install-recommends \
     build-essential \
     curl \
  && rm -rf /var/lib/apt/lists/*

# Install Poetry (installer via pip)
RUN pip install "poetry==$POETRY_VERSION"

# Copy only dependency definitions first to leverage Docker layer caching
COPY pyproject.toml poetry.lock* /app/

# Tell Poetry to install into the system environment rather than creating virtualenvs,
# then install dependencies (no dev packages) quietly.
RUN poetry config virtualenvs.create false \
  && poetry install --no-dev --no-interaction --no-ansi

# Copy the rest of the application code (including src/, api/, models/ if present)
COPY . /app

# Create a non-root user and give ownership of the app directory (safer than running as root)
RUN useradd --create-home appuser \
  && chown -R appuser:appuser /app

# Switch to the non-root user
USER appuser

# Expose the port that the app will run on
EXPOSE 8000

# Command to run the API using Uvicorn (change to Gunicorn for production-grade setups)
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]