from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class Vehicle(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    license_plate = db.Column(db.String(20), index=True)
    vehicle_type = db.Column(db.String(50))
    make = db.Column(db.String(50))
    model = db.Column(db.String(50))
    color = db.Column(db.String(20))
    confidence = db.Column(db.Float)
    location = db.Column(db.String(100))
    image_path = db.Column(db.String(200))
    
    def to_dict(self):
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat(),
            'license_plate': self.license_plate,
            'vehicle_type': self.vehicle_type,
            'make': self.make,
            'model': self.model,
            'color': self.color,
            'confidence': self.confidence,
            'location': self.location,
            'image_path': self.image_path
        }

class BlacklistedVehicle(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    license_plate = db.Column(db.String(20), unique=True, index=True)
    reason = db.Column(db.String(200))
    added_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'license_plate': self.license_plate,
            'reason': self.reason,
            'added_at': self.added_at.isoformat()
        }

class Alert(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    vehicle_id = db.Column(db.Integer, db.ForeignKey('vehicle.id'))
    alert_type = db.Column(db.String(50))
    message = db.Column(db.String(200))
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    is_read = db.Column(db.Boolean, default=False)
    
    def to_dict(self):
        return {
            'id': self.id,
            'vehicle_id': self.vehicle_id,
            'alert_type': self.alert_type,
            'message': self.message,
            'timestamp': self.timestamp.isoformat(),
            'is_read': self.is_read
        } 