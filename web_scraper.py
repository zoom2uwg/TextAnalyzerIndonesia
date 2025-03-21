"""
Web scraper for fetching Indonesian news and content from popular websites
to provide dynamic sample texts for analysis.
"""

import requests
from bs4 import BeautifulSoup
import random
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# List of Indonesian news sources
NEWS_SOURCES = [
    {
        "name": "Kompas",
        "url": "https://www.kompas.com/",
        "article_selector": ".article__link",
        "content_selector": ".read__content"
    },
    {
        "name": "Detik",
        "url": "https://www.detik.com/",
        "article_selector": "article h2 a",
        "content_selector": ".detail__body-text"
    },
    {
        "name": "Tempo",
        "url": "https://www.tempo.co/",
        "article_selector": ".title a",
        "content_selector": ".detail-konten"
    }
]

# Reviews source
REVIEW_SOURCES = [
    {
        "name": "ReviewGadget",
        "url": "https://www.tabloidpulsa.co.id/review/",
        "article_selector": ".td-module-title a",
        "content_selector": ".td-post-content"
    }
]

def get_random_news_article():
    """
    Fetch a random news article from one of the Indonesian news sources.
    
    Returns:
        tuple: (title, content, source_name) or (None, None, None) if failed
    """
    # Randomly select a news source
    source = random.choice(NEWS_SOURCES)
    
    try:
        # Fetch the main page
        logger.info(f"Fetching main page from {source['name']}")
        response = requests.get(source['url'], headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
        response.raise_for_status()
        
        # Parse the HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find article links
        article_links = soup.select(source['article_selector'])
        
        if not article_links:
            logger.warning(f"No article links found on {source['name']}")
            return None, None, None
        
        # Select a random article
        random_article = random.choice(article_links)
        article_url = random_article.get('href')
        
        if article_url and not article_url.startswith('http'):
            article_url = source['url'] + article_url
        
        # Fetch the article content
        if article_url and isinstance(article_url, str):
            logger.info(f"Fetching article from {article_url}")
            article_response = requests.get(article_url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
            article_response.raise_for_status()
        else:
            logger.warning("Invalid article URL")
            return None, None, None
        
        # Parse the article
        article_soup = BeautifulSoup(article_response.text, 'html.parser')
        
        # Get the title
        title = article_soup.title.string if article_soup.title else "Untitled"
        
        # Get the content
        content_element = article_soup.select_one(source['content_selector'])
        
        if content_element:
            # Extract text and clean it
            paragraphs = content_element.find_all('p')
            content = ' '.join([p.get_text() for p in paragraphs])
            
            # Trim to reasonable length (max 1000 characters)
            if len(content) > 1000:
                content = content[:1000] + "..."
                
            return title, content, source['name']
        else:
            logger.warning(f"Content not found in article from {source['name']}")
            return None, None, None
            
    except Exception as e:
        logger.error(f"Error fetching article from {source['name']}: {str(e)}")
        return None, None, None

def get_random_review():
    """
    Fetch a random product review from Indonesian review sources.
    
    Returns:
        tuple: (title, content, source_name) or (None, None, None) if failed
    """
    # Similar implementation as get_random_news_article but for reviews
    source = random.choice(REVIEW_SOURCES)
    
    try:
        # Fetch the main page
        logger.info(f"Fetching reviews from {source['name']}")
        response = requests.get(source['url'], headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
        response.raise_for_status()
        
        # Parse the HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find article links
        review_links = soup.select(source['article_selector'])
        
        if not review_links:
            logger.warning(f"No review links found on {source['name']}")
            return None, None, None
        
        # Select a random review
        random_review = random.choice(review_links)
        review_url = random_review.get('href')
        
        # Fetch the review content
        if review_url and isinstance(review_url, str):
            logger.info(f"Fetching review from {review_url}")
            review_response = requests.get(review_url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
            review_response.raise_for_status()
        else:
            logger.warning("Invalid review URL")
            return None, None, None
        
        # Parse the review
        review_soup = BeautifulSoup(review_response.text, 'html.parser')
        
        # Get the title
        title = review_soup.title.string if review_soup.title else "Untitled Review"
        
        # Get the content
        content_element = review_soup.select_one(source['content_selector'])
        
        if content_element:
            # Extract text and clean it
            paragraphs = content_element.find_all('p')
            content = ' '.join([p.get_text() for p in paragraphs])
            
            # Trim to reasonable length (max 1000 characters)
            if len(content) > 1000:
                content = content[:1000] + "..."
                
            return title, content, source['name']
        else:
            logger.warning(f"Content not found in review from {source['name']}")
            return None, None, None
            
    except Exception as e:
        logger.error(f"Error fetching review from {source['name']}: {str(e)}")
        return None, None, None

def get_dynamic_samples(num_samples=3):
    """
    Get a collection of dynamic samples from different sources.
    
    Args:
        num_samples (int): Number of samples to retrieve
        
    Returns:
        dict: Dictionary of sample name -> sample text
    """
    samples = {}
    
    # Try to get at least one news article
    for _ in range(3):  # Try 3 times
        title, content, source = get_random_news_article()
        if content:
            sample_name = f"Berita ({source}): {title}"
            samples[sample_name] = content
            break
    
    # Try to get a review if we need more samples
    if len(samples) < num_samples:
        for _ in range(3):  # Try 3 times
            title, content, source = get_random_review()
            if content:
                sample_name = f"Ulasan ({source}): {title}"
                samples[sample_name] = content
                break
    
    # Add some static backup samples if we couldn't get enough dynamic ones
    if len(samples) < num_samples:
        static_samples = {
            "Artikel Berita": "Jakarta - Presiden Joko Widodo mengatakan bahwa pembangunan infrastruktur merupakan kunci untuk meningkatkan konektivitas di Indonesia. Dalam kunjungannya ke Sulawesi Tengah, Presiden meresmikan jalan tol baru yang menghubungkan beberapa kabupaten di daerah tersebut. \"Infrastruktur adalah pondasi dari pertumbuhan ekonomi,\" kata Presiden. Menteri Pekerjaan Umum dan Perumahan Rakyat menambahkan bahwa pembangunan infrastruktur akan terus menjadi prioritas pemerintah. Diharapkan dengan adanya jalan tol baru ini, distribusi barang dan jasa akan semakin efisien dan dapat menurunkan biaya logistik.",
            "Ulasan Produk": "Hari ini saya akan mengulas smartphone terbaru dari Xiaomi. Dengan harga yang terjangkau, ponsel ini menawarkan spesifikasi yang mengagumkan. Layar AMOLED 6,5 inci sangat jernih dan responsif. Kamera utama 64MP menghasilkan foto yang tajam dengan warna yang akurat, meskipun di kondisi cahaya rendah masih ada sedikit noise. Baterai 5000mAh bertahan seharian penuh bahkan dengan penggunaan berat. Pengisian daya cepat 33W juga menjadi nilai plus. Secara keseluruhan, saya sangat puas dengan performa ponsel ini dan sangat merekomendasikannya untuk Anda yang mencari smartphone dengan harga terjangkau tetapi fitur premium.",
            "Artikel Ilmiah": "Penelitian terbaru tentang perkembangan kecerdasan buatan di Indonesia menunjukkan tren yang menggembirakan. Berdasarkan data dari Kementerian Riset dan Teknologi, adopsi AI di sektor industri telah meningkat sebesar 45% dalam dua tahun terakhir. Studi yang dilakukan oleh Universitas Indonesia mengidentifikasi beberapa faktor pendorong utama, termasuk dukungan pemerintah melalui insentif pajak dan program pelatihan. Namun, tantangan masih ada, terutama dalam hal infrastruktur digital dan kesenjangan keterampilan. Peneliti merekomendasikan peningkatan kerjasama antara akademisi, industri, dan pemerintah untuk mengatasi hambatan ini."
        }
        
        # Add static samples if needed
        for name, text in static_samples.items():
            if len(samples) < num_samples:
                if name not in samples:
                    samples[name] = text
            else:
                break
    
    return samples

if __name__ == "__main__":
    # Test the scraper
    samples = get_dynamic_samples(3)
    for name, content in samples.items():
        print(f"Sample: {name}")
        print(f"Length: {len(content)} characters")
        print(content[:150] + "...\n")