import React from 'react';
import '../css/Contacts.css';

const Contacts = () => {
  return (
    <div  className="contacts-sidebar hide" >
      <div className="contacts-header">
        <h3 className="contacts-title">Contacts</h3>
        <div className="contact-icons">
          <button className="icon-button">
            <i className="fas fa-video"></i>
          </button>
          <button className="icon-button">
            <i className="fas fa-search"></i>
          </button>
          <button className="icon-button">
            <i className="fas fa-ellipsis-h"></i>
          </button>
        </div>
      </div>

    
      <div className="contact-item">
        <img 
          src="https://randomuser.me/api/portraits/men/32.jpg" 
          alt="Mubara Khan" 
          className="contact-avatar"
        />
        <div>
          <div className="contact-name">Mubara Khan</div>
          <div className="contact-status">
            Active now
            <span className="online-dot"></span>
          </div>
        </div>
      </div>

      <div className="contact-item">
        <img 
          src="https://randomuser.me/api/portraits/women/44.jpg" 
          alt="Bushra Gulzar" 
          className="contact-avatar"
        />
        <div>
          <div className="contact-name">Bushra Gulzar</div>
          <div className="contact-status">20 minutes ago</div>
        </div>
      </div>

      <div className="contact-item">
        <img 
          src="https://randomuser.me/api/portraits/men/22.jpg" 
          alt="Rehafi Mughal" 
          className="contact-avatar"
        />
        <div>
          <div className="contact-name">Rehafi Mughal</div>
          <div className="contact-status">
            Active now
            <span className="online-dot"></span>
          </div>
        </div>
      </div>

      <div className="contact-item">
        <img 
          src="https://randomuser.me/api/portraits/men/45.jpg" 
          alt="Prasanta Ghosh" 
          className="contact-avatar"
        />
        <div>
          <div className="contact-name">Prasanta Ghosh</div>
          <div className="contact-status">1 hour ago</div>
        </div>
      </div>

      <div className="contact-item">
        <img 
          src="https://randomuser.me/api/portraits/women/33.jpg" 
          alt="Ayesha Tariq" 
          className="contact-avatar"
        />
        <div>
          <div className="contact-name">Ayesha Tariq</div>
          <div className="contact-status">Yesterday</div>
        </div>
      </div>

      <div className="contact-item">
        <img 
          src="https://randomuser.me/api/portraits/women/28.jpg" 
          alt="Eman Fatima" 
          className="contact-avatar"
        />
        <div>
          <div className="contact-name">Eman Fatima</div>
          <div className="contact-status">
            Active now
            <span className="online-dot"></span>
          </div>
        </div>
      </div>

      <div className="contact-item">
        <img 
          src="https://randomuser.me/api/portraits/women/19.jpg" 
          alt="Husna Asif" 
          className="contact-avatar"
        />
        <div>
          <div className="contact-name">Husna Asif</div>
          <div className="contact-status">2 days ago</div>
        </div>
      </div>

      <div className="contact-item">
        <img 
          src="https://randomuser.me/api/portraits/men/12.jpg" 
          alt="Ali Raza" 
          className="contact-avatar"
        />
        <div>
          <div className="contact-name">Ali Raza</div>
          <div className="contact-status">Online</div>
        </div>
      </div>
    </div>
  );
};

export default Contacts;