FROM python:3.10-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements_hf.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application files
COPY app.py .
COPY best_model.pt .

# Copy GIF animations
COPY steady_workouts.gif .
COPY intervals_workouts.gif .
COPY progressive_workouts.gif .
COPY category_comparison.gif .

# Copy JSON data files
COPY sample_steady.json .
COPY sample_intervals.json .
COPY sample_progressive.json .

# Expose Streamlit port
EXPOSE 7860

# Set environment variables for Streamlit
ENV STREAMLIT_SERVER_PORT=7860
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true

# Run Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
