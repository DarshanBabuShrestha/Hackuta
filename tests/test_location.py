import asyncio
import json
from pathlib import Path

from backend.dashboard import Dashboard


def test_location_endpoint_requires_key_and_valid_payload():
    async def check():
        dashboard = Dashboard(None, 'COM9', caregiver_key='secret')
        route = next(route for route in dashboard.app.routes
                     if getattr(route, 'path', None) == '/api/location')

        class Request:
            headers = {'x-caregiver-key': 'secret'}
            url = type('URL', (), {'hostname': 'localhost', 'scheme': 'http'})()

            async def json(self):
                return {'latitude': 32.7, 'longitude': -97.1, 'accuracy': 12,
                        'timestamp': '2026-11-15T12:00:00Z'}

        result = await route.endpoint(Request())
        assert result == {'ok': True}
        assert dashboard.snapshot(caregiver=True)['location']['latitude'] == 32.7
        assert 'location' not in dashboard.snapshot(caregiver=False)

        class BadRequest(Request):
            headers = {'x-caregiver-key': 'wrong'}

        denied = await route.endpoint(BadRequest())
        assert denied.status_code == 401
    asyncio.run(check())


def test_nonlocal_location_requires_https():
    async def check():
        dashboard = Dashboard(None, 'COM9', caregiver_key='secret')
        route = next(route for route in dashboard.app.routes
                     if getattr(route, 'path', None) == '/api/location')
        class Request:
            headers = {'x-caregiver-key': 'secret'}
            url = type('URL', (), {'hostname': 'wearer.example', 'scheme': 'http'})()
            async def json(self):
                return {'latitude': 1, 'longitude': 2, 'accuracy': 5,
                        'timestamp': '2026-11-15T12:00:00Z'}
        assert (await route.endpoint(Request())).status_code == 403
    asyncio.run(check())


def test_caregiver_map_uses_leaflet_osm_and_updates_marker():
    html = (Path(__file__).parents[1] / 'backend' / 'caregiver.html').read_text(encoding='utf-8')
    assert 'leaflet@1.9.4/dist/leaflet.js' in html
    assert 'tile.openstreetmap.org' in html
    assert 'OpenStreetMap contributors' in html
    assert 'wearerMarker.setLatLng(position)' in html
    assert 'accuracyCircle.setLatLng(position).setRadius(location.accuracy)' in html
    assert 'map-error' in html
