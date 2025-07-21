import logging
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time

from db import insert_car

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("app_activity.log"),
                        logging.StreamHandler()
                    ])
logger = logging.getLogger(__name__)

def scrape_autoplius(car_brands: list, pages_per_brand: int = 1, start_page: int = 1):
    """
    Scrapes car listing data from autoplius.lt for specified car brands and pages.

    Args:
        car_brands (list): A list of car brand slugs (e.g., ['bmw', 'audi']).
        pages_per_brand (int): The number of pages to scrape for each brand.
        start_page (int): The starting page number for scraping.
    """
    options = Options()
    driver = webdriver.Chrome(options=options)

    logger.info("Starting Autoplius.lt scraping process.")

    for brand in car_brands:
        logger.info(f"Starting data collection for brand: {brand}")
        for page in range(start_page, start_page + pages_per_brand):
            url = f"https://autoplius.lt/skelbimai/naudoti-automobiliai/{brand}?category_id=2&page_nr={page}"
            logger.info(f"Loading page: {url}")

            try:
                driver.get(url)
                time.sleep(5)
            except Exception as e:
                logger.error(f"Failed to load page {url}: {e}")
                continue

            soup = BeautifulSoup(driver.page_source, 'html.parser')
            listings = soup.find_all('a', {'class': 'announcement-item'})

            if not listings:
                logger.warning(f"No listings found on page {page} for brand '{brand}'. Ending collection for this brand.")
                break

            for listing in listings:
                try:
                    # Extract ad_id, which is critical for unique identification in the database.
                    ad_id = None
                    id_div = listing.find('div', class_='announcement-bookmark-button')
                    if id_div and id_div.has_attr('data-id'):
                        ad_id = id_div['data-id']

                    if not ad_id:
                        logger.warning("Could not find 'ad_id' for a listing. Skipping this listing.")
                        continue

                    # Extract car make and model from the main title element.
                    title_elem = listing.find('div', {'class': "announcement-title"})
                    if not title_elem:
                        logger.warning(f"Title element not found for ad_id: {ad_id}. Skipping.")
                        continue

                    title = title_elem.text.strip()
                    make_and_model = title.split(' ', 1)
                    make = make_and_model[0]
                    model = make_and_model[1] if len(make_and_model) > 1 else None

                    # Extract and clean the price value.
                    price_elem = listing.find('div', {'class': "announcement-pricing-info"})
                    price_value = None
                    if price_elem:
                        price_text = price_elem.text.strip()
                        price_cleaned = price_text.split('\n')[0].strip().replace('€', '').replace(' ', '')
                        try:
                            price_value = int(price_cleaned)
                        except ValueError:
                            logger.warning(f"Could not convert price '{price_cleaned}' to int for ad_id: {ad_id}")
                            price_value = None

                    # Extract year and body type from specific span elements.
                    year = None
                    body_type = None
                    title_params = listing.find('div', class_='announcement-title-parameters')
                    if title_params:
                        title_spans = title_params.find_all('span')
                        if len(title_spans) > 0:
                            year_text = title_spans[0].get_text(strip=True)[:4]
                            try:
                                year = int(year_text)
                            except ValueError:
                                logger.warning(f"Could not convert year '{year_text}' to int for ad_id: {ad_id}")
                                year = None
                        if len(title_spans) > 1:
                            body_type = title_spans[1].get_text(strip=True)

                    # Extract fuel type, gearbox, engine information, and mileage.
                    fuel = None
                    gearbox = None
                    engine_volume = None
                    engine_power = None
                    mileage = None
                    params_block = listing.find('div', class_='announcement-parameters-block')
                    if params_block:
                        block_spans = params_block.find_all('span')
                        block_spans_values = [s.get_text(strip=True) for s in block_spans]

                        if len(block_spans_values) > 0:
                            fuel = block_spans_values[0]
                        if len(block_spans_values) > 1:
                            gearbox = block_spans_values[1]
                        if len(block_spans_values) > 2:
                            engine_info = block_spans_values[2]
                            if ',' in engine_info:
                                parts = [p.strip() for p in engine_info.split(',')]
                                if len(parts) >= 2:
                                    volume_cleaned = parts[0].replace('l.', '').strip()
                                    try:
                                        engine_volume = float(volume_cleaned)
                                    except ValueError:
                                        logger.warning(f"Could not convert engine volume '{volume_cleaned}' to float for ad_id: {ad_id}")
                                        engine_volume = None

                                    power_cleaned = parts[1].replace('kW', '').strip()
                                    try:
                                        engine_power = int(power_cleaned)
                                    except ValueError:
                                        logger.warning(f"Could not convert engine power '{power_cleaned}' to int for ad_id: {ad_id}")
                                        engine_power = None
                            else:
                                logger.warning(f"Engine info format missing comma for '{engine_info}' for ad_id: {ad_id}")
                        if len(block_spans_values) > 3:
                            mileage_text = block_spans_values[3]
                            if ' km' in mileage_text.lower():
                                mileage_cleaned = mileage_text.lower().replace('km', '').replace(' ', '').strip()
                                try:
                                    mileage = int(mileage_cleaned)
                                except ValueError:
                                    logger.warning(f"Could not convert mileage '{mileage_cleaned}' to int for ad_id: {ad_id}")
                                    mileage = None
                            else:
                                logger.warning(f"Mileage format not recognized or 'km' not found for '{mileage_text}' for ad_id: {ad_id}")

                    car_data = (ad_id, make, model, price_value, year, body_type, fuel, gearbox, engine_volume, engine_power, mileage)
                    insert_car(car_data)
                    logger.info(f"Successfully inserted car listing (ID: {ad_id}) into DB.")
                except Exception as e:
                    logger.error(f"An unexpected error occurred while processing a listing: {e}. Skipping this listing. Ad ID might be missing or unidentifiable.", exc_info=True)
                    continue

    driver.quit()
    logger.info("Autoplius.lt scraping process completed.")

if __name__ == "__main__":
    car_brands = [
        'bmw',
        'volkswagen',
        'audi',
        'mercedes_benz',
        'toyota',
        'volvo',
        'opel',
        'ford',
        'peugeot',
        'skoda'
    ]
    # Adjust pages_per_brand to scrape more data (e.g., 5 or 10 pages).
    scrape_autoplius(car_brands, pages_per_brand=1, start_page=1)
    logger.info("web_scrapper.py script execution finished.")
