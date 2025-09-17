import requests
import json
import logging
from typing import Dict, Any, Optional
from app.config import settings
import hashlib
import hmac
import time

logger = logging.getLogger(__name__)


class WebhookNotifier:
    """Webhook notification system for EKYC events"""
    
    def __init__(self):
        self.webhook_url = settings.webhook_url
        self.webhook_secret = settings.webhook_secret
        self.timeout = settings.webhook_timeout
    
    def _generate_signature(self, payload: str) -> str:
        """Generate HMAC signature for webhook payload"""
        if not self.webhook_secret:
            return ""
        
        signature = hmac.new(
            self.webhook_secret.encode('utf-8'),
            payload.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        return f"sha256={signature}"
    
    def _send_webhook(self, event_type: str, data: Dict[str, Any]) -> bool:
        """Send webhook notification"""
        if not self.webhook_url:
            logger.info("Webhook URL not configured, skipping notification")
            return True
        
        try:
            payload = {
                "event": event_type,
                "timestamp": int(time.time()),
                "data": data
            }
            
            payload_json = json.dumps(payload, ensure_ascii=False)
            
            headers = {
                "Content-Type": "application/json",
                "User-Agent": f"EKYC-Service/{settings.version}",
                "X-EKYC-Event": event_type
            }
            
            # Add signature if secret is configured
            if self.webhook_secret:
                signature = self._generate_signature(payload_json)
                headers["X-EKYC-Signature"] = signature
            
            response = requests.post(
                self.webhook_url,
                data=payload_json,
                headers=headers,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                logger.info(f"Webhook sent successfully for event: {event_type}")
                return True
            else:
                logger.warning(f"Webhook failed with status {response.status_code}: {response.text}")
                return False
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Webhook request failed: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error sending webhook: {str(e)}")
            return False
    
    def notify_session_created(self, session_id: str, user_id: str) -> bool:
        """Notify when EKYC session is created"""
        data = {
            "session_id": session_id,
            "user_id": user_id,
            "status": "created"
        }
        return self._send_webhook("session.created", data)
    
    def notify_session_completed(
        self, 
        session_id: str, 
        user_id: str, 
        final_decision: str,
        face_match_score: Optional[float] = None,
        liveness_score: Optional[float] = None
    ) -> bool:
        """Notify when EKYC session is completed"""
        data = {
            "session_id": session_id,
            "user_id": user_id,
            "status": "completed",
            "final_decision": final_decision,
            "scores": {
                "face_match": face_match_score,
                "liveness": liveness_score
            }
        }
        return self._send_webhook("session.completed", data)
    
    def notify_session_failed(
        self, 
        session_id: str, 
        user_id: str, 
        error_message: str
    ) -> bool:
        """Notify when EKYC session fails"""
        data = {
            "session_id": session_id,
            "user_id": user_id,
            "status": "failed",
            "error_message": error_message
        }
        return self._send_webhook("session.failed", data)
    
    def notify_asset_uploaded(
        self, 
        session_id: str, 
        asset_id: str, 
        asset_type: str,
        user_id: str
    ) -> bool:
        """Notify when asset is uploaded"""
        data = {
            "session_id": session_id,
            "asset_id": asset_id,
            "asset_type": asset_type,
            "user_id": user_id,
            "status": "uploaded"
        }
        return self._send_webhook("asset.uploaded", data)
    
    def notify_asset_processed(
        self, 
        session_id: str, 
        asset_id: str, 
        asset_type: str,
        user_id: str,
        success: bool,
        processing_result: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Notify when asset processing is completed"""
        data = {
            "session_id": session_id,
            "asset_id": asset_id,
            "asset_type": asset_type,
            "user_id": user_id,
            "status": "processed",
            "success": success,
            "result": processing_result
        }
        return self._send_webhook("asset.processed", data)
    
    def notify_face_match_completed(
        self, 
        session_id: str, 
        user_id: str,
        similarity_score: float,
        is_match: bool
    ) -> bool:
        """Notify when face matching is completed"""
        data = {
            "session_id": session_id,
            "user_id": user_id,
            "face_match": {
                "similarity_score": similarity_score,
                "is_match": is_match,
                "threshold": settings.face_match_threshold
            }
        }
        return self._send_webhook("face_match.completed", data)


# Global webhook notifier instance
webhook_notifier = WebhookNotifier()
