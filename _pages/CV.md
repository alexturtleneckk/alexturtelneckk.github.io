---
layout: default
title: CV
permalink: /categories/CV.html
---

<div class="container">
    <div class="row justify-content-center">
        <div class="col-md-8">
            <h1 class="font-weight-bold title h6 text-uppercase mb-4">{{ page.title }}</h1>
            
            {% for post in site.categories.CV %}
                {% include main-loop-card.html %}
            {% endfor %}
        </div>
        
        <div class="col-md-4">
            {% include sidebar-featured.html %}    
        </div>
    </div>
</div>