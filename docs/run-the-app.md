Here are the instructions to start and test the backend application.

  1. Starting the Backend Services


  The project uses Docker Compose to manage all the necessary services. This is the most reliable way to run the backend for
  development and testing.

  Step 1: Configure Environment Variables

  The application requires API keys and other configuration to be set as environment variables.

   * Create a .env file by copying the provided template:


   1     cp .env.example .env

   * Open the new .env file in a text editor.
   * Fill in the required values, especially API keys for the LLM providers you intend to use (e.g., OPENAI_API_KEY).

  Step 2: Launch the Services


   * Open your terminal in the project's root directory.
   * Run the following command:

   1     docker-compose up -d

   * This command will:
       * Build the Docker image for the chatbot API.
       * Start all the services defined in docker-compose.yml in the background (-d for detached mode). This includes the main
         FastAPI application, a PostgreSQL database, a Redis cache, and a Chroma vector database.
       * The API will be running and accessible at `http://localhost:8000`.


  2. Testing the Backend

  You can test the backend in a few different ways:

  Method 1: Simple Health Check

   * Once the containers are running, you can send a request to the health check endpoint to verify the API is responsive. Open
     a new terminal and use curl:


   1     curl http://localhost:8000/health

   * You should receive a response like {"status": "ok"}.

  Method 2: Using Postman


   * With the services running, you can use Postman to interact with any of the API endpoints.
   * The main chat endpoint is likely available at POST /api/v1/chat/message (based on the project structure).
   * To test it:
       1. Open Postman and create a new POST request.
       2. Set the URL to http://localhost:8000/api/v1/chat/message.
       3. Go to the Body tab, select raw, and choose JSON from the dropdown.
       4. Enter a JSON payload like this:


   1         {
   2           "message": "Hello, world!",
   3           "sessionId": "postman-test-session"
   4         }

       5. Click Send. You should receive a chatbot response.

  Method 3: Running the Automated Test Suite


  The project has a full suite of automated tests that you can run.


   * To execute the tests, run this command from your project's root directory:

   1     docker-compose exec chatbot-api pytest

   * This command runs the pytest test suite inside the running chatbot-api container, which ensures the tests have access to
     the database and other services they need to run correctly.