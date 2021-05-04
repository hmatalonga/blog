<template>
  <header class="flex justify-between pb-4 mb-8 md:pb-0 md:mb-16 xl:mb-24">
    <div class="font-semibold text-gray-800 uppercase lg:text-base xl:text-lg">
      <nuxt-link class="logo" to="/">Hugo Matalonga</nuxt-link>
    </div>
    <div>
      <div
        class="flex items-center space-x-6 lg:space-x-8 lg:text-base xl:text-lg text-gray-600"
      >
        <div class="space-x-6 lg:space-x-8">
          <nuxt-link class="hover:text-gray-800" to="/work-journal"
            >Work journal</nuxt-link
          >
          <nuxt-link class="hover:text-gray-800" to="/blog">Blog</nuxt-link>
        </div>
        <div>
          <a href="/feed.xml">
            <svg
              class="current-color w-5 h-5"
              xmlns="http://www.w3.org/2000/svg"
              viewBox="0 0 20 20"
              fill="currentColor"
            >
              <path
                d="M5 3a1 1 0 000 2c5.523 0 10 4.477 10 10a1 1 0 102 0C17 8.373 11.627 3 5 3z"
              />
              <path
                d="M4 9a1 1 0 011-1 7 7 0 017 7 1 1 0 11-2 0 5 5 0 00-5-5 1 1 0 01-1-1zM3 15a2 2 0 114 0 2 2 0 01-4 0z"
              />
            </svg>
          </a>
        </div>
      </div>
    </div>
  </header>
</template>
