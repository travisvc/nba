import requests
from typing import List, Dict, Optional
from loguru import logger


class PolymarketGammaAPI:
    def __init__(self, base_url: str = "https://gamma-api.polymarket.com"):
        self.base_url = base_url
        logger.info("Initializing Gamma API client...")
        logger.success("Gamma API client initialized successfully")
    
    def get_series_list(self) -> List[Dict]:
        try:
            url = f"{self.base_url}/series"
            logger.info("Fetching all available series...")
            
            response = requests.get(url)
            response.raise_for_status()
            series = response.json()
            
            logger.success(f"Successfully fetched {len(series)} series")
            return series
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching series list: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error fetching series list: {e}")
            raise
    
    def get_series_by_id(self, series_id: int, include_closed: bool = False) -> Dict:
        try:
            url = f"{self.base_url}/series/{series_id}"
            params = {"closed": "false"} if not include_closed else {}
            
            logger.info(f"Fetching series {series_id} (include_closed={include_closed})...")
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            series_data = response.json()
            

            events = series_data.get('events', [])
            if not include_closed:
                events = [
                    e for e in events 
                    if e.get('active', False) and not e.get('closed', False) and not e.get('archived', False)
                ]
            
            events_with_markets = []
            for event in events:
                try:

                    event_url = f"{self.base_url}/events/{event['id']}"
                    event_response = requests.get(event_url)
                    event_response.raise_for_status()
                    event_data = event_response.json()
                    
                    event_copy = event.copy()
                    event_copy['markets'] = event_data.get('markets', [])
                    events_with_markets.append(event_copy)
                    
                except Exception as e:
                    logger.warning(f"Failed to fetch markets for event {event['id']}: {e}")

                    event_copy = event.copy()
                    event_copy['markets'] = []
                    events_with_markets.append(event_copy)
            
            result = {
                'series': series_data,
                'events': events_with_markets,
                'total_events': len(events_with_markets),
                'total_markets': sum(len(event.get('markets', [])) for event in events_with_markets)
            }
            
            logger.success(f"Successfully fetched series {series_id}: {len(events_with_markets)} events, {result['total_markets']} markets")
            return result
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching series {series_id}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error fetching series {series_id}: {e}")
            raise
    
    def get_markets_by_series_id(self, series_id: int, include_closed: bool = False, limit: Optional[int] = None) -> List[Dict]:
        try:
            series_data = self.get_series_by_id(series_id, include_closed)
            events = series_data.get('events', [])
            
            all_markets = []
            for event in events:
                markets = event.get('markets', [])
                all_markets.extend(markets)
            
            if limit:
                all_markets = all_markets[:limit]
            
            logger.success(f"Retrieved {len(all_markets)} markets from {len(events)} events in series {series_id}")
            return all_markets
            
        except Exception as e:
            logger.error(f"Error fetching markets for series {series_id}: {e}")
            raise
    
    def get_events_by_series_id(self, series_id: int, include_closed: bool = False, limit: Optional[int] = None) -> List[Dict]:
        try:
            series_data = self.get_series_by_id(series_id, include_closed)
            events = series_data.get('events', [])
            
            if limit:
                events = events[:limit]
            
            logger.success(f"Retrieved {len(events)} events from series {series_id}")
            return events
            
        except Exception as e:
            logger.error(f"Error fetching events for series {series_id}: {e}")
            raise
    
    def get_market_details(self, market_id: int) -> Dict:
        try:
            url = f"{self.base_url}/events/{market_id}"
            logger.info(f"Fetching market details for ID {market_id}...")
            
            response = requests.get(url)
            response.raise_for_status()
            market_data = response.json()
            
            logger.success(f"Successfully fetched market details for ID {market_id}")
            return market_data
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching market {market_id}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error fetching market {market_id}: {e}")
            raise
    
    def get_series_summary(self, series_id: int) -> Dict:
        try:
            series_data = self.get_series_by_id(series_id, include_closed=True)
            events = series_data['events']
            active_events = series_data['active_events']
            
            total_volume = sum(float(e.get('volume', 0)) for e in events)
            active_volume = sum(float(e.get('volume', 0)) for e in active_events)
            
            categories = list(set(e.get('category', 'Unknown') for e in events))
            
            summary = {
                'series_info': series_data['series_info'],
                'statistics': {
                    'total_events': len(events),
                    'active_events': len(active_events),
                    'closed_events': len(events) - len(active_events),
                    'total_volume': total_volume,
                    'active_volume': active_volume,
                    'categories': categories
                }
            }
            
            logger.success(f"Generated summary for series {series_id}")
            return summary
            
        except Exception as e:
            logger.error(f"Error generating summary for series {series_id}: {e}")
            raise



