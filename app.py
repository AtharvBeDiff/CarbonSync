import os
import datetime
from typing import List, Optional

from fastapi import FastAPI, Depends, HTTPException, status, APIRouter, UploadFile, File, Form
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr
from sqlalchemy import create_engine, Column, Integer, String, Text, TIMESTAMP, ForeignKey, DECIMAL, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from sqlalchemy.sql import func
import uuid

# --- Configuration ---
# In a real app, use environment variables
DATABASE_URL = "sqlite:///./carbon_credit_marketplace.db"
SECRET_KEY = "a_very_secret_key_that_should_be_in_env_vars"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# --- Database Setup (database.py) ---
engine = create_engine(
    DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- Password Hashing Setup ---
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# --- Models (models.py) ---
class Organization(Base):
    __tablename__ = "Organizations"
    organization_id = Column(Integer, primary_key=True, index=True)
    organization_name = Column(String(255), nullable=False)
    organization_type = Column(String(50), nullable=False)
    address = Column(Text)
    contact_info = Column(Text)
    verification_status = Column(String(50), nullable=False, default="Pending Verification")
    created_at = Column(TIMESTAMP, server_default=func.now())

class User(Base):
    __tablename__ = "Users"
    user_id = Column(Integer, primary_key=True, index=True)
    organization_id = Column(Integer, ForeignKey("Organizations.organization_id"))
    email = Column(String(255), nullable=False, unique=True, index=True)
    password_hash = Column(String(255), nullable=False)
    full_name = Column(String(255), nullable=False)
    role = Column(String(50), nullable=False) # e.g., 'Admin', 'Project Developer', 'Investor', 'Verifier'
    verification_status = Column(String(50), nullable=False, default="Pending Verification")
    wallet_address = Column(String(255), nullable=False, unique=True)
    created_at = Column(TIMESTAMP, server_default=func.now())
    organization = relationship("Organization")

class Project(Base):
    __tablename__ = "Projects"
    project_id = Column(Integer, primary_key=True, index=True)
    organization_id = Column(Integer, ForeignKey("Organizations.organization_id"), nullable=False)
    project_name = Column(String(255), nullable=False)
    description = Column(Text)
    ecosystem_type = Column(String(100), nullable=False)
    status = Column(String(50), nullable=False, default="Pending Approval")
    location_boundary = Column(Text, nullable=False) # Simplified GEOMETRY to TEXT
    created_at = Column(TIMESTAMP, server_default=func.now())
    organization = relationship("Organization")

class ProjectDocument(Base):
    __tablename__ = "ProjectDocuments"
    document_id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey("Projects.project_id"), nullable=False)
    document_type = Column(String(100), nullable=False)
    file_url = Column(String(255), nullable=False)
    uploaded_at = Column(TIMESTAMP, server_default=func.now())
    project = relationship("Project")

class DataSubmission(Base):
    __tablename__ = "DataSubmissions"
    submission_id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey("Projects.project_id"), nullable=False)
    submitted_by_user_id = Column(Integer, ForeignKey("Users.user_id"), nullable=False)
    submission_date = Column(TIMESTAMP, server_default=func.now())
    status = Column(String(50), nullable=False, default="Pending Verification")
    project = relationship("Project")
    submitter = relationship("User")

class EvidenceItem(Base):
    __tablename__ = "EvidenceItems"
    evidence_id = Column(Integer, primary_key=True, index=True)
    submission_id = Column(Integer, ForeignKey("DataSubmissions.submission_id"), nullable=False)
    evidence_type = Column(String(50), nullable=False)
    file_url = Column(String(255))
    note_text = Column(Text)
    geotag = Column(Text, nullable=False) # Simplified GEOMETRY to TEXT
    timestamp = Column(TIMESTAMP, nullable=False)
    submission = relationship("DataSubmission")

class Verification(Base):
    __tablename__ = "Verifications"
    verification_id = Column(Integer, primary_key=True, index=True)
    submission_id = Column(Integer, ForeignKey("DataSubmissions.submission_id"), nullable=False, unique=True)
    verifier_id = Column(Integer, ForeignKey("Users.user_id"), nullable=False)
    decision = Column(String(50), nullable=False) # 'Approved', 'Rejected'
    comments = Column(Text)
    verification_date = Column(TIMESTAMP, server_default=func.now())
    submission = relationship("DataSubmission")
    verifier = relationship("User")

class CarbonCredit(Base):
    __tablename__ = "CarbonCredits"
    credit_id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey("Projects.project_id"), nullable=False)
    verification_id = Column(Integer, ForeignKey("Verifications.verification_id"), nullable=False)
    owner_organization_id = Column(Integer, ForeignKey("Organizations.organization_id"), nullable=False)
    quantity = Column(DECIMAL(18, 4), nullable=False)
    vintage_year = Column(Integer, nullable=False)
    minting_date = Column(TIMESTAMP, server_default=func.now())
    blockchain_transaction_hash = Column(String(255), nullable=False, unique=True)
    status = Column(String(50), nullable=False, default="Available") # 'Available', 'Listed', 'Sold'
    project = relationship("Project")
    verification = relationship("Verification")
    owner = relationship("Organization")

class MarketplaceListing(Base):
    __tablename__ = "MarketplaceListings"
    listing_id = Column(Integer, primary_key=True, index=True)
    credit_id = Column(Integer, ForeignKey("CarbonCredits.credit_id"), nullable=False)
    seller_organization_id = Column(Integer, ForeignKey("Organizations.organization_id"), nullable=False)
    quantity_for_sale = Column(DECIMAL(18, 4), nullable=False)
    price_per_credit = Column(DECIMAL(18, 2), nullable=False)
    status = Column(String(50), nullable=False, default="Active") # 'Active', 'Sold', 'Cancelled'
    listing_date = Column(TIMESTAMP, server_default=func.now())
    credit = relationship("CarbonCredit")
    seller = relationship("Organization")

class Transaction(Base):
    __tablename__ = "Transactions"
    transaction_id = Column(Integer, primary_key=True, index=True)
    listing_id = Column(Integer, ForeignKey("MarketplaceListings.listing_id"), nullable=False)
    buyer_user_id = Column(Integer, ForeignKey("Users.user_id"), nullable=False)
    seller_organization_id = Column(Integer, ForeignKey("Organizations.organization_id"), nullable=False)
    quantity_purchased = Column(DECIMAL(18, 4), nullable=False)
    total_price = Column(DECIMAL(18, 2), nullable=False)
    transaction_date = Column(TIMESTAMP, server_default=func.now())
    blockchain_transaction_hash = Column(String(255), nullable=False, unique=True)
    listing = relationship("MarketplaceListing")
    buyer = relationship("User")
    seller = relationship("Organization")

# --- Schemas (schemas.py) ---

# Token
class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    email: Optional[str] = None

# User Schemas
class UserBase(BaseModel):
    email: EmailStr
    full_name: str

class UserCreate(UserBase):
    password: str
    role: str
    organization_name: str
    organization_type: str
    address: str
    contact_info: str

class UserRegisterResponse(BaseModel):
    user_id: int
    email: EmailStr
    full_name: str
    role: str
    verification_status: str
    message: str

class UserLoginResponse(BaseModel):
    accessToken: str
    user_id: int
    role: str

class UserInDB(UserBase):
    user_id: int
    organization_id: int
    role: str
    verification_status: str
    wallet_address: str
    created_at: datetime.datetime
    class Config:
        orm_mode = True

class UserUpdate(BaseModel):
    full_name: Optional[str] = None
    contact_info: Optional[str] = None

# Organization Schemas
class OrganizationStatusUpdate(BaseModel):
    verification_status: str # 'Approved', 'Rejected'
    comments: Optional[str] = None

class OnboardingQueueItem(BaseModel):
    organization_id: int
    organization_name: str
    organization_type: str
    verification_status: str
    created_at: datetime.datetime
    class Config:
        orm_mode = True

# Project Schemas
class ProjectCreate(BaseModel):
    project_name: str
    description: str
    ecosystem_type: str
    location_boundary: str

class ProjectResponse(BaseModel):
    project_id: int
    organization_id: int
    project_name: str
    status: str
    created_at: datetime.datetime
    class Config:
        orm_mode = True

class ProjectDetail(ProjectResponse):
    description: str
    ecosystem_type: str
    location_boundary: str

class ProjectList(BaseModel):
    project_id: int
    project_name: str
    description: str
    ecosystem_type: str
    status: str
    class Config:
        orm_mode = True

class ProjectStatusUpdate(BaseModel):
    status: str
    comments: Optional[str] = None

# Document Schema
class DocumentResponse(BaseModel):
    document_id: int
    project_id: int
    document_type: str
    file_url: str
    uploaded_at: datetime.datetime
    class Config:
        orm_mode = True

# Submission Schemas
class SubmissionResponse(BaseModel):
    submission_id: int
    project_id: int
    submitted_by_user_id: int
    submission_date: datetime.datetime
    status: str
    class Config:
        orm_mode = True

class EvidenceItemResponse(BaseModel):
    evidence_id: int
    submission_id: int
    evidence_type: str
    file_url: Optional[str]
    geotag: str
    timestamp: datetime.datetime
    class Config:
        orm_mode = True

class SubmissionQueueItem(BaseModel):
    submission_id: int
    project_id: int
    project_name: str
    submission_date: datetime.datetime
    status: str

class SubmissionDetail(SubmissionResponse):
    evidence_items: List[EvidenceItemResponse]

class VerificationCreate(BaseModel):
    decision: str
    comments: Optional[str] = None

class VerificationResponse(BaseModel):
    verification_id: int
    submission_id: int
    decision: str
    message: str

# Wallet/Credit Schemas
class CreditResponse(BaseModel):
    credit_id: int
    project_id: int
    quantity: float
    vintage_year: int
    minting_date: datetime.datetime
    status: str
    class Config:
        orm_mode = True

class TransactionResponse(BaseModel):
    transaction_id: int
    listing_id: int
    quantity_purchased: float
    total_price: float
    transaction_date: datetime.datetime
    blockchain_transaction_hash: str
    type: str # 'buy' or 'sell'
    class Config:
        orm_mode = True

# Marketplace Schemas
class MarketplaceProject(BaseModel):
    project_id: int
    project_name: str
    description: str
    ecosystem_type: str
    location_boundary: str
    class Config:
        orm_mode = True

class MarketplaceProjectDetail(MarketplaceProject):
    organization_name: str
    total_credits_minted: float

class MarketplaceListingResponse(BaseModel):
    listing_id: int
    credit_id: int
    seller_organization_name: str
    quantity_for_sale: float
    price_per_credit: float
    vintage_year: int
    project_name: str

class ListingCreate(BaseModel):
    credit_id: int
    quantity_for_sale: float
    price_per_credit: float

class ListingCreateResponse(BaseModel):
    listing_id: int
    status: str
    message: str

class BuyRequest(BaseModel):
    listing_id: int
    quantity_to_purchase: float

class BuyResponse(BaseModel):
    transaction_id: int
    total_price: float
    blockchain_transaction_hash: str
    message: str

# --- Security (auth.py) ---
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[datetime.timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.datetime.utcnow() + expires_delta
    else:
        expire = datetime.datetime.utcnow() + datetime.timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def get_user(db: Session, email: str):
    return db.query(User).filter(User.email == email).first()

def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
        token_data = TokenData(email=email)
    except JWTError:
        raise credentials_exception
    user = get_user(db, email=token_data.email)
    if user is None:
        raise credentials_exception
    return user

# Role-based access control dependencies
def require_role(required_role: str):
    def role_checker(current_user: User = Depends(get_current_user)):
        if current_user.role != required_role:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=f"Operation not permitted for user with role '{current_user.role}'")
        return current_user
    return role_checker

require_admin = require_role("Admin")
require_project_developer = require_role("Project Developer")
require_investor = require_role("Investor")
require_verifier = require_role("Verifier")

# --- Routers --- 
auth_router = APIRouter()
users_router = APIRouter()
admin_router = APIRouter()
projects_router = APIRouter()
submissions_router = APIRouter()
wallet_router = APIRouter()
marketplace_router = APIRouter()

# --- Auth Router (/api/auth) ---
@auth_router.post("/register", response_model=UserRegisterResponse)
def register_user(user: UserCreate, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.email == user.email).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")

    # Create Organization
    new_org = Organization(
        organization_name=user.organization_name,
        organization_type=user.organization_type,
        address=user.address,
        contact_info=user.contact_info
    )
    db.add(new_org)
    db.commit()
    db.refresh(new_org)

    # Create User
    hashed_password = get_password_hash(user.password)
    new_user = User(
        full_name=user.full_name,
        email=user.email,
        password_hash=hashed_password,
        role=user.role,
        organization_id=new_org.organization_id,
        wallet_address=f'0x{uuid.uuid4().hex}' # Dummy wallet address
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    
    return {
        "user_id": new_user.user_id,
        "email": new_user.email,
        "full_name": new_user.full_name,
        "role": new_user.role,
        "verification_status": new_user.verification_status,
        "message": "Registration successful. Your account is pending verification."
    }

@auth_router.post("/login", response_model=UserLoginResponse)
def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = get_user(db, email=form_data.username)
    if not user or not verify_password(form_data.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if user.verification_status != "Approved":
        raise HTTPException(status_code=403, detail="User account is not approved yet.")

    access_token_expires = datetime.timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.email}, expires_delta=access_token_expires
    )
    return {"accessToken": access_token, "user_id": user.user_id, "role": user.role}

# --- Users Router (/api/users) ---
@users_router.get("/me", response_model=UserInDB)
def read_users_me(current_user: User = Depends(get_current_user)):
    return current_user

# --- Admin Router (/api/admin) ---
@admin_router.get("/onboarding-queue", response_model=List[OnboardingQueueItem])
def get_onboarding_queue(db: Session = Depends(get_db), admin_user: User = Depends(require_admin)):
    organizations = db.query(Organization).filter(Organization.verification_status == "Pending Verification").all()
    return organizations

@admin_router.put("/organizations/{organizationId}/status")
def update_organization_status(organizationId: int, status_update: OrganizationStatusUpdate, db: Session = Depends(get_db), admin_user: User = Depends(require_admin)):
    org = db.query(Organization).filter(Organization.organization_id == organizationId).first()
    if not org:
        raise HTTPException(status_code=404, detail="Organization not found")
    
    org.verification_status = status_update.verification_status
    
    # Approve all users in the organization as well
    if status_update.verification_status == "Approved":
        db.query(User).filter(User.organization_id == organizationId).update({"verification_status": "Approved"})

    db.commit()
    return {"organization_id": org.organization_id, "organization_name": org.organization_name, "verification_status": org.verification_status, "message": "Organization status updated successfully."}

@admin_router.put("/projects/{projectId}/status")
def update_project_status(projectId: int, status_update: ProjectStatusUpdate, db: Session = Depends(get_db), admin_user: User = Depends(require_admin)):
    project = db.query(Project).filter(Project.project_id == projectId).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    project.status = status_update.status
    db.commit()
    return {"project_id": project.project_id, "status": project.status, "message": "Project status updated successfully."}

@admin_router.get("/submissions-queue", response_model=List[SubmissionQueueItem])
def get_submissions_queue(db: Session = Depends(get_db), verifier: User = Depends(require_verifier)):
    submissions = db.query(DataSubmission).join(Project).filter(DataSubmission.status == "Pending Verification").all()
    result = []
    for sub in submissions:
        result.append(SubmissionQueueItem(
            submission_id=sub.submission_id,
            project_id=sub.project_id,
            project_name=sub.project.project_name,
            submission_date=sub.submission_date,
            status=sub.status
        ))
    return result

# --- Projects Router (/api/projects) ---
@projects_router.post("", response_model=ProjectResponse)
def create_project(project: ProjectCreate, db: Session = Depends(get_db), dev_user: User = Depends(require_project_developer)):
    new_project = Project(**project.dict(), organization_id=dev_user.organization_id)
    db.add(new_project)
    db.commit()
    db.refresh(new_project)
    return new_project

@projects_router.get("/organizations/{organizationId}/projects", response_model=List[ProjectList])
def get_organization_projects(organizationId: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    # Ensure user is part of the organization or an admin
    if current_user.organization_id != organizationId and current_user.role != 'Admin':
        raise HTTPException(status_code=403, detail="Not authorized to view these projects")
    projects = db.query(Project).filter(Project.organization_id == organizationId).all()
    return projects

@projects_router.get("/{projectId}", response_model=ProjectDetail)
def get_project_details(projectId: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    project = db.query(Project).filter(Project.project_id == projectId).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return project

@projects_router.post("/{projectId}/documents", response_model=DocumentResponse)
async def upload_project_document(projectId: int, document_type: str = Form(...), file: UploadFile = File(...), db: Session = Depends(get_db), dev_user: User = Depends(require_project_developer)):
    project = db.query(Project).filter(Project.project_id == projectId).first()
    if not project or project.organization_id != dev_user.organization_id:
        raise HTTPException(status_code=403, detail="Not authorized for this project")
    
    # In a real app, upload to S3/GCS and get URL
    file_url = f"/uploads/documents/{uuid.uuid4()}_{file.filename}"

    new_doc = ProjectDocument(project_id=projectId, document_type=document_type, file_url=file_url)
    db.add(new_doc)
    db.commit()
    db.refresh(new_doc)
    return new_doc

@projects_router.post("/{projectId}/submissions", response_model=SubmissionResponse)
def create_data_submission(projectId: int, db: Session = Depends(get_db), dev_user: User = Depends(require_project_developer)):
    project = db.query(Project).filter(Project.project_id == projectId).first()
    if not project or project.organization_id != dev_user.organization_id:
        raise HTTPException(status_code=403, detail="Not authorized for this project")

    new_submission = DataSubmission(project_id=projectId, submitted_by_user_id=dev_user.user_id)
    db.add(new_submission)
    db.commit()
    db.refresh(new_submission)
    return new_submission

# --- Submissions Router (/api/submissions) ---
@submissions_router.post("/{submissionId}/evidence", response_model=EvidenceItemResponse)
async def upload_evidence(submissionId: int, evidence_type: str = Form(...), geotag: str = Form(...), timestamp: datetime.datetime = Form(...), file: Optional[UploadFile] = File(None), note_text: Optional[str] = Form(None), db: Session = Depends(get_db), dev_user: User = Depends(require_project_developer)):
    submission = db.query(DataSubmission).filter(DataSubmission.submission_id == submissionId).first()
    if not submission or submission.submitted_by_user_id != dev_user.user_id:
        raise HTTPException(status_code=403, detail="Not authorized for this submission")

    file_url = None
    if file:
        file_url = f"/uploads/evidence/{uuid.uuid4()}_{file.filename}"

    new_evidence = EvidenceItem(
        submission_id=submissionId,
        evidence_type=evidence_type,
        file_url=file_url,
        note_text=note_text,
        geotag=geotag,
        timestamp=timestamp
    )
    db.add(new_evidence)
    db.commit()
    db.refresh(new_evidence)
    return new_evidence

@submissions_router.get("/{submissionId}", response_model=SubmissionDetail)
def get_submission_details(submissionId: int, db: Session = Depends(get_db), verifier: User = Depends(require_verifier)):
    submission = db.query(DataSubmission).filter(DataSubmission.submission_id == submissionId).first()
    if not submission:
        raise HTTPException(status_code=404, detail="Submission not found")
    
    evidence = db.query(EvidenceItem).filter(EvidenceItem.submission_id == submissionId).all()
    
    return SubmissionDetail(
        submission_id=submission.submission_id,
        project_id=submission.project_id,
        submission_date=submission.submission_date,
        status=submission.status,
        submitted_by_user_id=submission.submitted_by_user_id,
        evidence_items=evidence
    )

@submissions_router.post("/{submissionId}/verify", response_model=VerificationResponse)
def verify_submission(submissionId: int, verification_data: VerificationCreate, db: Session = Depends(get_db), verifier: User = Depends(require_verifier)):
    submission = db.query(DataSubmission).filter(DataSubmission.submission_id == submissionId).first()
    if not submission:
        raise HTTPException(status_code=404, detail="Submission not found")
    if submission.status != "Pending Verification":
        raise HTTPException(status_code=400, detail="Submission already verified")
    
    new_verification = Verification(
        submission_id=submissionId,
        verifier_id=verifier.user_id,
        decision=verification_data.decision,
        comments=verification_data.comments
    )
    db.add(new_verification)

    submission.status = verification_data.decision
    db.commit()
    db.refresh(new_verification)
    
    message = "Submission verified successfully."
    if verification_data.decision == "Approved":
        # Automated minting process
        project = submission.project
        new_credit = CarbonCredit(
            project_id=project.project_id,
            verification_id=new_verification.verification_id,
            owner_organization_id=project.organization_id,
            quantity=1000.0, # Dummy quantity
            vintage_year=datetime.datetime.now().year,
            blockchain_transaction_hash=f"0x{uuid.uuid4().hex}"
        )
        db.add(new_credit)
        db.commit()
        message = "Submission approved and carbon credits minted successfully."

    return {"verification_id": new_verification.verification_id, "submission_id": submissionId, "decision": new_verification.decision, "message": message}

# --- Wallet Router (/api/wallet) ---
@wallet_router.get("/credits", response_model=List[CreditResponse])
def get_wallet_credits(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    credits = db.query(CarbonCredit).filter(CarbonCredit.owner_organization_id == current_user.organization_id).all()
    return credits

# --- Marketplace Router (/api/marketplace) ---
@marketplace_router.get("/listings", response_model=List[MarketplaceListingResponse])
def get_all_listings(db: Session = Depends(get_db)):
    listings = db.query(MarketplaceListing).filter(MarketplaceListing.status == "Active").all()
    response = []
    for l in listings:
        response.append(MarketplaceListingResponse(
            listing_id=l.listing_id,
            credit_id=l.credit_id,
            seller_organization_name=l.seller.organization_name,
            quantity_for_sale=l.quantity_for_sale,
            price_per_credit=l.price_per_credit,
            vintage_year=l.credit.vintage_year,
            project_name=l.credit.project.project_name
        ))
    return response

@marketplace_router.post("/listings", response_model=ListingCreateResponse)
def create_listing(listing_data: ListingCreate, db: Session = Depends(get_db), dev_user: User = Depends(require_project_developer)):
    credit = db.query(CarbonCredit).filter(CarbonCredit.credit_id == listing_data.credit_id).first()
    if not credit or credit.owner_organization_id != dev_user.organization_id:
        raise HTTPException(status_code=403, detail="Not authorized to sell these credits")
    if credit.status != "Available":
        raise HTTPException(status_code=400, detail="Credits are not available for sale")
    if credit.quantity < listing_data.quantity_for_sale:
        raise HTTPException(status_code=400, detail="Not enough credits to sell")

    new_listing = MarketplaceListing(
        credit_id=listing_data.credit_id,
        seller_organization_id=dev_user.organization_id,
        quantity_for_sale=listing_data.quantity_for_sale,
        price_per_credit=listing_data.price_per_credit
    )
    credit.status = "Listed"
    db.add(new_listing)
    db.commit()
    db.refresh(new_listing)

    return {"listing_id": new_listing.listing_id, "status": new_listing.status, "message": "Credits listed successfully."}

@marketplace_router.post("/transactions/buy", response_model=BuyResponse)
def buy_credits(buy_data: BuyRequest, db: Session = Depends(get_db), investor: User = Depends(require_investor)):
    listing = db.query(MarketplaceListing).filter(MarketplaceListing.listing_id == buy_data.listing_id).first()
    if not listing or listing.status != "Active":
        raise HTTPException(status_code=404, detail="Listing not found or is not active")
    if listing.quantity_for_sale < buy_data.quantity_to_purchase:
        raise HTTPException(status_code=400, detail="Not enough credits available in this listing")

    total_price = buy_data.quantity_to_purchase * listing.price_per_credit

    # In a real app, this would involve a payment gateway and smart contract interaction
    # Here we simulate the transaction

    # Create the transaction record
    new_transaction = Transaction(
        listing_id=buy_data.listing_id,
        buyer_user_id=investor.user_id,
        seller_organization_id=listing.seller_organization_id,
        quantity_purchased=buy_data.quantity_to_purchase,
        total_price=total_price,
        blockchain_transaction_hash=f"0x{uuid.uuid4().hex}"
    )
    db.add(new_transaction)

    # Update listing quantity
    listing.quantity_for_sale -= buy_data.quantity_to_purchase
    if listing.quantity_for_sale == 0:
        listing.status = "Sold"

    # Update original credit ownership
    credit = listing.credit
    credit.quantity -= buy_data.quantity_to_purchase

    # Create new credit for buyer
    buyer_credit = CarbonCredit(
        project_id=credit.project_id,
        verification_id=credit.verification_id,
        owner_organization_id=investor.organization_id,
        quantity=buy_data.quantity_to_purchase,
        vintage_year=credit.vintage_year,
        blockchain_transaction_hash=f"0x{uuid.uuid4().hex}",
        status="Available"
    )
    db.add(buyer_credit)

    db.commit()
    db.refresh(new_transaction)

    return {"transaction_id": new_transaction.transaction_id, "total_price": total_price, "blockchain_transaction_hash": new_transaction.blockchain_transaction_hash, "message": "Purchase successful."}


# --- Main Application (main.py) ---
app = FastAPI(title="Carbon Credit Marketplace API")

@app.on_event("startup")
def on_startup():
    Base.metadata.create_all(bind=engine)

app.include_router(auth_router, prefix="/api/auth", tags=["Authentication"])
app.include_router(users_router, prefix="/api/users", tags=["Users"])
app.include_router(admin_router, prefix="/api/admin", tags=["Admin"])
app.include_router(projects_router, prefix="/api/projects", tags=["Projects"])
app.include_router(submissions_router, prefix="/api/submissions", tags=["Submissions & Verification"])
app.include_router(wallet_router, prefix="/api/wallet", tags=["Wallet"])
app.include_router(marketplace_router, prefix="/api/marketplace", tags=["Marketplace"])

@app.get("/", tags=["Root"])
def read_root():
    return {"message": "Welcome to the Carbon Credit Marketplace API"}

# To run this app: `uvicorn main:app --reload` (if you save this as main.py)
# Then access the docs at http://127.0.0.1:8000/docs