# README

## Bookmark Clustering Application

This application is designed to cluster and organize bookmarks extracted from web browsers into meaningful groups. It uses natural language processing and machine learning algorithms to analyze bookmark titles and URLs, categorize bookmarks into clusters, and generate descriptive cluster names.

### Input Data Format

The input data is expected to be a JSON list of dictionaries. Each dictionary represents a bookmark with the following keys:

- `name`: The title of the bookmark.
- `url`: The web address the bookmark points to.
- `tags`: A list of tags associated with the bookmark.

Example input data:
```json
[
  {
    "name": "Mozilla Support",
    "url": "https://support.mozilla.org/products/firefox",
    "tags": ["Bookmarks Menu", "Mozilla Firefox"]
  },
  {
    "name": "OpenAI Chat",
    "url": "https://chat.openai.com/?model=gpt-4",
    "tags": ["Bookmarks Toolbar"]
  },
  ...
]
```

### Output Data Format

The output data is a JSON dictionary where keys are generated cluster names and the values are lists of bookmark details that fall within the corresponding cluster.

Example output data:
```json
{
  "Tech Resource Hub": [
    {
      "name": "Mozilla Support",
      "url": "https://support.mozilla.org/products/firefox"
    },
    {
      "name": "OpenAI Chat",
      "url": "https://chat.openai.com/?model=gpt-4"
    }
  ],
  "Data Science Community": [
    ...
  ],
  ...
}
```

### API Endpoint

The main endpoint `/cluster` expects a POST request with a JSON payload containing bookmark data. It processes the data and responds with the clustered bookmarks.

### Usage Instructions

1. Send a POST request to `/cluster` with the JSON data as a request body.
2. Receive a response with the clustered bookmarks in JSON format.

Ensure that your server is running and accessible for the API endpoint to be available.

### Development and Testing

To work on or test the application, follow these steps:

1. Ensure you have Python and Flask installed on your local machine.
2. Navigate to the application directory in your terminal or command prompt.
3. Run `python run.py` to start the development server.
4. Use tools like `curl` or Postman to send POST requests to `http://localhost:5000/cluster`.