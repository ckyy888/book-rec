import requests

def check_google_books_ratings(title):
    """Check how Google Books provides ratings information"""
    query = f'intitle:{title}'
    url = f'https://www.googleapis.com/books/v1/volumes?q={query}'
    
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        print("\nRaw API response for ratings data:")
        if 'items' in data:
            book = data['items'][0]
            volume_info = book.get('volumeInfo', {})
            print(f"Title: {volume_info.get('title')}")
            print(f"Ratings data: {volume_info.get('ratingsCount')}")
            print(f"Average rating: {volume_info.get('averageRating')}")
            print("\nFull volume info:")
            for key, value in volume_info.items():
                print(f"{key}: {value}")
    else:
        print(f"Error: {response.status_code}")

# Test with a popular book
check_google_books_ratings("American Gods") 